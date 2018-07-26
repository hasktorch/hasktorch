{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE BangPatterns #-}
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

import qualified Codec.Picture as JP
import qualified Graphics.GD as GD



#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
import Control.Concurrent (threadDelay)
#else
import Torch.Double
import qualified Torch.Long as Long
#endif

-- This should be replaced with a download-aware cache.
default_cifar_path :: FilePath
default_cifar_path = "/mnt/lake/datasets/cifar-10"

data Mode = Test | Train
  deriving (Eq, Enum, Ord, Show, Bounded)

testLength  :: Proxy 'Test -> Proxy 1000
testLength _ = Proxy

trainLength :: Proxy 'Train -> Proxy 5000
trainLength _ = Proxy

data Categories
  = Airplane
  | Automobile
  | Bird
  | Cat
  | Deer
  | Dog
  | Frog
  | Horse
  | Ship
  | Truck
  deriving (Eq, Enum, Ord, Show, Bounded)

category_path :: FilePath -> Mode -> Categories -> FilePath
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
  im <- fromFoldable (Data.List.take 20 ims)

  lift (runExceptT (im2torch (fp </> im))) >>= \case
    Left _ -> mempty
    Right t -> do
      pure t

oneHot :: KnownDim n => Int -> IO (LongTensor '[n])
oneHot i = do
  let c = Long.constant 0
  Long._set1d c (fromIntegral i) 1
  pure c

cifar10set :: FilePath -> Mode -> ListT IO (Tensor '[3, 32, 32], Integer)
cifar10set fp m = do
  c <- fromFoldable [minBound..maxBound :: Categories]
  t <- cat2torch (category_path fp m c)

  pure (t, fromIntegral (fromEnum c))

defaultCifar10set :: Mode -> ListT IO (Tensor '[3, 32, 32], Integer)
defaultCifar10set = cifar10set default_cifar_path


#ifdef DEBUG
-- example use-case
main :: IO ()
main = do
  forM_ [minBound..maxBound :: Mode] $ \m -> do
    l <- toList (defaultCifar10set m)
--  print $ length l
#endif

