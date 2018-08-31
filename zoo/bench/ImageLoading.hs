{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
module ImageLoading (img_loading_bench) where

import Criterion.Main
import Criterion.Types (resamples)

import GHC.TypeLits
import Control.Monad
import Control.Monad.Trans.Class
import Control.Monad.Trans.Except
import Control.Monad.Primitive
import Data.Vector (Vector)
import Data.Vector.Mutable (MVector)
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as M
import qualified Codec.Picture as JP

#ifdef CUDA
import Torch.Cuda.Double
#else
import Torch.Double
#endif


img_loading_bench :: [Benchmark]
img_loading_bench = [
  bgroup "Image Loading on CIFAR-10"
    [ bench "rgb2torch_setter"  $ nfIO (go rgb2torch_setter >>= tensordata)
    , bench "rgb2torch_storage" $ nfIO (go rgb2torch_storage >>= tensordata)
    ]
  ]
 where
  file = "/mnt/lake/datasets/cifar-10/train/cat/1_cat.png"

  go :: (FilePath -> ExceptT String IO (Tensor '[3, 32, 32])) -> IO (Tensor '[3, 32, 32])
  go getter = runExceptT (getter file) >>=
    \case
      Left s -> error "could not load"
      Right t -> pure t


-- | load an RGB PNG image into a Torch tensor
rgb2torch_setter
  :: forall h w . (KnownDim h, KnownDim w)
  => FilePath -> ExceptT String IO (Tensor '[3, h, w])
rgb2torch_setter fp = do
  t <- lift new
  im <- JP.convertRGB8 <$> ExceptT (JP.readPng fp)
  setPixels $ \(h, w) -> do
    let JP.PixelRGB8 r g b = JP.pixelAt im h w
    forM_ (zip [0..] [r, g, b]) $ \(c, px) ->
      setDim'_ t (someDimsVal $ fromIntegral <$> [c, h, w]) (fromIntegral px)
  pure t
 where
  setPixels :: ((Int, Int) -> IO ()) -> ExceptT String IO ()
  setPixels
    = lift . forM_ [(h, w) | h <- [0.. fromIntegral (dimVal (dim :: Dim h)) - 1]
                           , w <- [0.. fromIntegral (dimVal (dim :: Dim w)) - 1]]


-- | load an RGB PNG image into a Torch tensor
rgb2torch_storage
  :: forall h w . (All KnownDim '[h, w], All KnownNat '[h, w])
  -- => Normalize
  => FilePath
  -> ExceptT String IO (Tensor '[3, h, w])
rgb2torch_storage {-(Normalize norm)-} fp = do
  im <- JP.convertRGB8 <$> ExceptT (JP.readPng fp)
  ExceptT $ do
    vec <- mkRGBVec
    fillFrom im vec
    cuboid <$> freezeList vec
 where
  prep w =
    if True -- norm
    then fromIntegral w {-/ 255 -}
    else fromIntegral w

  height :: Int
  height = fromIntegral (dimVal (dim :: Dim h))

  width :: Int
  width = fromIntegral (dimVal (dim :: Dim w))

  mkRGBVec :: PrimMonad m => m (MRGBVector (PrimState m))
  mkRGBVec = M.replicateM 3 (M.replicateM height (M.unsafeNew width))

  writePx
    :: PrimMonad m
    => MRGBVector (PrimState m)
    -> (Int, Int, Int)
    -> HsReal
    -> m ()
  writePx channels (c, h, w) px
    = M.unsafeRead channels c
    >>= \rows -> M.unsafeRead rows h
    >>= \cols -> M.unsafeWrite cols w px

  fillFrom :: PrimMonad m => JP.Image JP.PixelRGB8 -> MRGBVector (PrimState m) -> m ()
  fillFrom im vec =
    forM_ [(h, w) | h <- [0.. height - 1], w <- [0.. width - 1]] $ \(h, w) -> do
      let JP.PixelRGB8 r g b = JP.pixelAt im h w
      forM_ (zip [0..] [r,g,b]) $ \(c, px) ->
        writePx vec (c, h, w) (prep px)

  freeze
    :: forall m s
    . PrimMonad m
    => s ~ PrimState m
    => MRGBVector s
    -> m (Vector (Vector (Vector HsReal)))
  freeze mvecs = do
    -- frames <-
    readNfreeze mvecs 3 $ \mframe ->
      -- forM [0,1,2] $ (M.read mvecs >=> \mframe ->
        readNfreeze mframe height V.unsafeFreeze
    -- pure $ V.fromListN 3 frames

  freezeList
    :: forall m s
    . PrimMonad m
    => s ~ PrimState m
    => MRGBVector s
    -> m [[[HsReal]]]
  freezeList mvecs = do
    readN mvecs 3 $ \mframe ->
      readN mframe height $ \mrow ->
        readN mrow width pure



readNfreeze :: PrimMonad m => MVector (PrimState m) a -> Int -> (a -> m b) -> m (Vector b)
readNfreeze mvec n op =
  V.fromListN n <$> readN mvec n op

readN :: PrimMonad m => MVector (PrimState m) a -> Int -> (a -> m b) -> m [b]
readN mvec n op = mapM (M.read mvec >=> op) [0..n-1]


type MRGBVector s = MVector s (MVector s (MVector s HsReal))
type RGBVector = Vector (Vector (Vector HsReal))


