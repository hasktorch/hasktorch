{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE CPP #-}
module Torch.Data.Loaders.Cifar10 where

import Control.Monad
import Data.Char
import Data.List
import System.Directory
import System.FilePath
import Control.Monad.Trans.Except
import Control.Monad.Trans.Class
import ListT
import Data.Singletons
import Numeric.Dimensions
import Data.Singletons.Prelude.Enum
import GHC.TypeLits
import Numeric.Backprop

import Text.Printf
import System.IO (hFlush, stdout)
import Data.IORef

import qualified Data.Vector as V
import qualified Codec.Picture as JP
import qualified Graphics.GD as GD

#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double
import qualified Torch.Long as Long
#endif

import qualified Torch.Double as CPU
import qualified Torch.Long as CPULong

-- This should be replaced with a download-aware cache.
default_cifar_path :: FilePath
default_cifar_path = "/mnt/lake/datasets/cifar-10"

data Mode = Test | Train
  deriving (Eq, Enum, Ord, Show, Bounded)

testLength  :: Proxy 'Test -> Proxy 1000
testLength _ = Proxy

trainLength :: Proxy 'Train -> Proxy 5000
trainLength _ = Proxy

data Category
  = Airplane    -- 0
  | Automobile  -- 2
  | Bird        -- 3
  | Cat         -- 4
  | Deer        -- 5
  | Dog         -- 6
  | Frog        -- 7
  | Horse       -- 8
  | Ship        -- 9
  | Truck       -- 10
  deriving (Eq, Enum, Ord, Show, Bounded)

category_path :: FilePath -> Mode -> Category -> FilePath
category_path cifarpath m c = intercalate "/"
  [ cifarpath
  , toLower <$> show m
  , toLower <$> show c
  ]

im2torch :: FilePath -> ExceptT String IO (Tensor '[3, 32, 32])
im2torch fp = do
  -- !im <- JP.convertRGB8 <$> ExceptT (JP.readPng fp)
  !im <- lift $ GD.loadPngFile fp
  !t <- lift new
  lift $ forM_ [(h, w) | h <- [0..31], w <- [0..31]] $ \(h, w) -> do
    -- let JP.PixelRGB8 r g b = JP.pixelAt im h w
    (r,g,b,_) <- GD.toRGBA <$> GD.getPixel (h,w) im
    forM_ (zip [0..] [r, g, b]) $ \(c, px) ->
      setDim'_ t (someDimsVal $ fromIntegral <$> [c, h, w]) (fromIntegral px)
  pure t

cat2torch :: FilePath -> ListT IO (Tensor '[3, 32, 32])
cat2torch fp = do
  ims <- lift $ filter ((== ".png") . takeExtension) <$> getDirectoryContents fp
  counter <- lift $ newIORef (0 :: Int)
  im <- fromFoldable ims -- (Data.List.take 20 ims)

  lift (runExceptT (im2torch (fp </> im))) >>= \case
    Left _ -> mempty
    Right t -> do
      lift $ do
        c <- (modifyIORef counter (+ 1) >> readIORef counter)
        printf "\r[%d/%d] img: %s" c (length ims) (fp </> im)
        hFlush stdout

      pure t

onehotL
  :: forall c sz
  . (Ord c, Bounded c, Enum c) -- , sz ~ FromEnum (MaxBound c), KnownDim sz, KnownNat sz)
  => c
  -> LongTensor '[10] -- '[FromEnum (MaxBound c)]
onehotL c
  = Long.unsafeVector
  $ onehot c

onehotT
  :: forall c sz
  . (Ord c, Bounded c, Enum c) -- , sz ~ FromEnum (MaxBound c), KnownDim sz, KnownNat sz)
  => c
  -> Tensor '[10] -- '[FromEnum (MaxBound c)]
onehotT c
  = unsafeVector
  $ fmap fromIntegral
  $ onehot c

onehot
  :: forall i c
  . (Integral i, Ord c, Bounded c, Enum c)
  => c
  -> [i]
onehot c
  = V.toList
  $ V.generate
    (fromEnum (maxBound :: c) + 1)
    (fromIntegral . fromEnum . (== fromEnum c))

onehotf
  :: forall i c
  . (Fractional i, Ord c, Bounded c, Enum c)
  => c
  -> [i]
onehotf c
  = V.toList
  $ V.generate
    (fromEnum (maxBound :: c) + 1)
    (realToFrac . fromIntegral . fromEnum . (== fromEnum c))



cifar10set :: FilePath -> Mode -> ListT IO (Tensor '[3, 32, 32], Category)
cifar10set fp m = do
  c <- fromFoldable [minBound..mx]
  lift $ printf "\n[%s](%d/%d)\n" (show c) (1 + fromEnum c) (1 + fromEnum mx)
  t <- cat2torch (category_path fp m c)
  pure (t, c)
 where
  mx :: Category
  mx = maxBound

defaultCifar10set :: Mode -> ListT IO (Tensor '[3, 32, 32], Category)
defaultCifar10set = cifar10set default_cifar_path

test :: Tensor '[1]
test
  = evalBP
      (classNLLCriterion (Long.unsafeVector [2] :: Long.Tensor '[1]))
      (unsqueeze1d (dim :: Dim 0) $ unsafeVector [1,0,0] :: Tensor '[1, 3])

-- test2 :: Tensor '[1]
-- test2
--   = evalBP
--   ( _classNLLCriterion'
--       (-100) False True
--       -- (Long.unsafeMatrix [[0,1,0]] :: Long.Tensor '[1,3])
--       -- (Long.unsafeVector [0,1,0] :: Long.Tensor '[3])
--       (Long.unsafeVector [0,1,2] :: Long.Tensor '[3])
--     )
--     -- (unsafeVector  [1,0,0]  :: Tensor '[3])
--     -- (unsafeMatrix [[0,0,1]] :: Tensor '[1,3])
--     (unsafeMatrix
--       [ [1,0,0]
--       , [0,1,0]
--       , [0.5,0.5,0.5]
--       ] :: Tensor '[3,3])

-- test3 :: CPU.Tensor '[1]
-- test3
--   = evalBP
--   ( CPU._classNLLCriterion'
--       (-100) False True
--       -- (CPULong.unsafeMatrix [[0,1,0]] :: CPULong.Tensor '[1,3])
--       (CPULong.unsafeVector [0,8] :: CPULong.Tensor '[2])
--       -- (CPULong.unsafeVector [0,1,0] :: CPULong.Tensor '[3])
--       -- (CPULong.unsafeVector [0,1,2] :: CPULong.Tensor '[3])
--     )
--     -- (CPU.unsafeVector  [1,0,0]  :: CPU.Tensor '[3])
--     -- (CPU.unsafeMatrix [[0,0,1]] :: CPU.Tensor '[1,3])
--     (CPU.unsafeMatrix
--       [ [1,0,0]
--       -- , [0,1,0]
--       , [0.5,0.5,0.5]
--       ] :: CPU.Tensor '[2,3])

#ifdef DEBUG
-- example use-case
main :: IO ()
main = do
  forM_ [minBound..maxBound :: Mode] $ \m -> do
    l <- toList (defaultCifar10set m)
    print $ length l
#endif

