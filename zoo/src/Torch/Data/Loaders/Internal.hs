{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TupleSections #-}
module Torch.Data.Loaders.Internal where

import Data.Proxy
import Data.Vector (Vector)
import Control.Concurrent (threadDelay)
import Control.Monad -- (forM_, filterM)
import Control.Monad.Trans.Class
import Control.Monad.Trans.Except
import GHC.Conc (getNumProcessors)
import GHC.TypeLits (KnownNat)
import Numeric.Dimensions
import System.Random.MWC (GenIO)
import System.Random.MWC.Distributions (uniformShuffle)
import System.Directory (listDirectory, doesDirectoryExist)
import System.FilePath ((</>), takeExtension)

import Control.Monad.Primitive
import qualified Data.Vector as V
import Data.Vector.Mutable (MVector)
import qualified Data.Vector.Mutable as M

import qualified Control.Concurrent.MSem as MSem (new, with)
import qualified Control.Concurrent.Async as Async

#ifdef USE_GD
import qualified Graphics.GD as GD
#else
import qualified Codec.Picture as JP
#endif

#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double
import qualified Torch.Long as Long
#endif


-- | asyncronously map across a pool with a maximum level of concurrency
mapPool :: Traversable t => Int -> (a -> IO b) -> t a -> IO (t b)
mapPool mx fn xs = do
  sem <- MSem.new mx
  Async.mapConcurrently (MSem.with sem . fn) xs

-- | load an RGB PNG image into a Torch tensor
rgb2torch
  :: forall h w
  . (All KnownDim '[h, w], All KnownNat '[h, w])
  => FilePath -> ExceptT String IO (Tensor '[3, h, w])
rgb2torch fp = do
  t <- lift new
  pxs <- file2rgb (Proxy :: Proxy '(h, w)) fp
  lift . fillFrom pxs $ \(c, h, w) px -> setDim'_ t (someDimsVal $ fromIntegral <$> [c, h, w]) px
  pure t

fillFrom :: (Num y, PrimMonad m) => [((Int, Int), (Int, Int, Int))] -> ((Int, Int, Int) -> y -> m ()) -> m ()
fillFrom pxs filler =
  forM_ pxs $ \((h, w), (r, g, b)) ->
    forM_ (zip [0..] [r,g,b]) $ \(c, px) ->
      filler (c, h, w) (fromIntegral px)

file2rgb
  :: forall h w hw rgb
  . (All KnownDim '[h, w], All KnownNat '[h, w])
  => hw ~ (Int, Int)
  => rgb ~ (Int, Int, Int)
  => Proxy '(h, w)
  -> FilePath
  -> ExceptT String IO [(hw, rgb)]
file2rgb hwp fp = do
#ifdef USE_GD
  im <- lift $ GD.loadPngFile fp
  forM [(h, w) | h <- [0.. height - 1], w <- [0.. width - 1]] $ \(h, w) -> do
    (r,g,b,_) <- lift $ GD.toRGBA <$> GD.getPixel (h,w) im
#else
  im <- JP.convertRGB8 <$> ExceptT (JP.readPng fp)
  forM [(h, w) | h <- [0.. height - 1], w <- [0.. width - 1]] $ \(h, w) -> do
    let JP.PixelRGB8 r g b = JP.pixelAt im h w
#endif
    pure ((h, w), (fromIntegral r, fromIntegral g, fromIntegral b))
 where
  (height, width) = reifyHW hwp

reifyHW
  :: forall h w
  . (All KnownDim '[h, w], All KnownNat '[h, w])
  => Proxy '(h, w)
  -> (Int, Int)
reifyHW _ = (fromIntegral (dimVal (dim :: Dim h)), fromIntegral (dimVal (dim :: Dim w)))


newtype Normalize = Normalize Bool
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | load an RGB PNG image into a Torch tensor
rgb2torch'
  :: forall h w . (All KnownDim '[h, w], All KnownNat '[h, w])
  => Normalize
  -> FilePath
  -> ExceptT String IO (Tensor '[3, h, w])
rgb2torch' (Normalize donorm) fp = do
  pxs <- file2rgb hwp fp
  ExceptT $ do
    vec <- mkRGBVec
    fillFrom pxs (\chw px -> writePx vec chw (prep px))
    cuboid <$> freezeList vec
 where
  prep w =
    if donorm
    then w / 255
    else w

  (height, width) = reifyHW hwp

  hwp :: Proxy '(h, w)
  hwp = Proxy

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


-- | Given a folder with subfolders of category images, return a uniform-randomly
-- shuffled list of absolute filepaths with the corresponding category.
shuffleCatFolders
  :: forall c
  .  GenIO                        -- ^ generator for shuffle
  -> (FilePath -> Maybe c)        -- ^ how to convert a subfolder into a category
  -> FilePath                     -- ^ absolute path of the dataset
  -> IO (Vector (c, FilePath))    -- ^ shuffled list
shuffleCatFolders g cast path = do
  cats <- filterM (doesDirectoryExist . (path </>)) =<< listDirectory path
  imgfiles <- sequence $ catContents <$> cats
  uniformShuffle (V.concat imgfiles) g
 where
  catContents :: FilePath -> IO (Vector (c, FilePath))
  catContents catFP =
    case cast catFP of
      Nothing -> pure mempty
      Just c ->
        let
          fdr = path </> catFP
          asPair img = (c, fdr </> img)
        in
          V.fromList . fmap asPair . filter isImage
          <$> listDirectory fdr

-- | verifies that an absolute filepath is an image
isImage :: FilePath -> Bool
isImage = (== ".png") . takeExtension

#ifdef DEBUG
main = do
  nprocs <- getNumProcessors
  putStrLn $ "number of cores: " ++ show nprocs
  mapPool nprocs (\x -> threadDelay 100000 >> print x >> pure x) [1..100]
#endif
