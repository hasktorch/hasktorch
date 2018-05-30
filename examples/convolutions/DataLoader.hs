{-# LANGUAGE LambdaCase #-}
module DataLoader where

import Codec.Picture
import Control.Monad
import Data.Char
import Data.List
import System.Directory
import System.FilePath
import Control.Monad.Trans.Except
import Control.Monad.Trans.Class
import ListT

import Torch.Double

-- This should be replaced with a download-aware cache.
cifar_path :: FilePath
cifar_path = "/mnt/lake/datasets/cifar-10"

data Mode = Test | Train
  deriving (Eq, Enum, Ord, Show, Bounded)

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
      someDimsM [c, h, w] >>= \d -> setDim'_ t d (fromIntegral px)
  pure t

cat2torch :: FilePath -> ListT IO (Tensor '[3, 32, 32])
cat2torch fp = do
  ims <- lift $ filter ((== ".png") . takeExtension) <$> getDirectoryContents fp
  im <- fromFoldable ims

  mt <- lift (runExceptT (im2torch (fp </> im)))
  case mt of
    Left _ -> mempty
    Right t -> pure t

cifar10set :: Mode -> ListT IO (Tensor '[3, 32, 32], Int)
cifar10set m = do
  c <- fromFoldable [minBound..maxBound :: Categories]
  t <- cat2torch (category_path m c)
  pure (t, fromEnum c)


