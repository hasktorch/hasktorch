{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DataKinds #-}
module DataLoader where

import Codec.Picture
import Control.Monad
import Data.Char
import Data.Maybe
import Data.List
import System.Directory
import System.FilePath
import Control.Monad.Trans.Except
import Control.Monad.Trans.Class
import ListT
import Data.Singletons

import Torch.Double
import qualified Torch.Long as Long

-- This should be replaced with a download-aware cache.
cifar_path :: FilePath
cifar_path = "/mnt/lake/datasets/cifar-10"

data Mode = Test | Train
  deriving (Eq, Enum, Ord, Show, Bounded)

testLength  :: Proxy 'Test -> Proxy 1000
testLength _ = undefined

trainLength :: Proxy 'Train -> Proxy 5000
trainLength _ = undefined

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

category_path :: Mode -> Categories -> FilePath
category_path m c = intercalate "/"
  [ cifar_path
  , toLower <$> show m
  , toLower <$> show c
  ]

main :: IO ()
main = do
  forM_ [minBound..maxBound :: Mode] $ \m -> do
    l <- toList (cifar10set m)
    print $ length l

im2torch :: FilePath -> ExceptT String IO (Tensor '[3, 32, 32])
im2torch fp = do
  dynIm <- ExceptT $ readPng fp
  let im = convertRGB8 dynIm
  t <- lift new
  lift $ forM_ [(h, w) | h <- [0..31], w <- [0..31]] $ \(h, w) -> do
    let PixelRGB8 r g b = pixelAt im h w
    forM_ (zip [0..] [r, g, b]) $ \(c, px) ->
      setDim'_ t (someDimsVal $ fromIntegral <$> [c, h, w]) (fromIntegral px)
  pure t

cat2torch :: FilePath -> ListT IO (Tensor '[3, 32, 32])
cat2torch fp = do
  ims <- lift $ filter ((== ".png") . takeExtension) <$> getDirectoryContents fp
  im <- fromFoldable ims

  mt <- lift (runExceptT (im2torch (fp </> im)))
  case mt of
    Left _ -> mempty
    Right t -> pure t

oneHot :: KnownDim n => Int -> IO (LongTensor '[n])
oneHot i = do
  let c = Long.constant 0
  Long._set1d c (fromIntegral i) 1
  pure c

cifar10set :: Mode -> ListT IO (Tensor '[3, 32, 32], Integer)
cifar10set m = do
  c <- fromFoldable [minBound..maxBound :: Categories]
  t <- cat2torch (category_path m c)
  pure (t, fromIntegral (fromEnum c))

