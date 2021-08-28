{-# LANGUAGE NoMonomorphismRestriction #-}

module VisionSpec (spec) where

import Codec.Picture
import Codec.Picture.Types (freezeImage, newMutableImage)
import Control.Exception.Safe
import Control.Monad.ST
import Data.Int
import Data.Word
import Test.Hspec
import Torch.DType
import Torch.Layout
import Torch.Tensor
import Torch.TensorFactories
import Torch.TensorOptions
import Torch.Vision

newImage :: Pixel a => Int -> Int -> [((Int, Int), a)] -> Image a
newImage width height ipixels = runST $ do
  mim <- newMutableImage height width
  mapM_ (\((x, y), rgb) -> writePixel mim x y rgb) ipixels
  freezeImage mim

spec :: Spec
spec = do
  describe "fromImage" $ do
    it "RGBA16" $ do
      let img = fromDynImage $ ImageRGBA16 $ newImage 1 3 (map (\i -> ((i, 0), PixelRGBA16 (fromIntegral i) 0 0 3)) [0 .. 2])
      (asValue img :: [[[[Int32]]]]) `shouldBe` [[[[0, 0, 0, 3], [1, 0, 0, 3], [2, 0, 0, 3]]]]
    it "RGB16" $ do
      let img = fromDynImage $ ImageRGB16 $ newImage 1 3 (map (\i -> ((i, 0), PixelRGB16 (fromIntegral i) 0 4)) [0 .. 2])
      (asValue img :: [[[[Int32]]]]) `shouldBe` [[[[0, 0, 4], [1, 0, 4], [2, 0, 4]]]]
    it "Y32" $ do
      let img = fromDynImage $ ImageY32 $ newImage 1 3 (map (\i -> ((i, 0), (fromIntegral i))) [0 .. 2])
      (asValue img :: [[[[Int64]]]]) `shouldBe` [[[[0], [1], [2]]]]
    it "Y16" $ do
      let img = fromDynImage $ ImageY16 $ newImage 1 3 (map (\i -> ((i, 0), (fromIntegral i))) [0 .. 2])
      (asValue img :: [[[[Int32]]]]) `shouldBe` [[[[0], [1], [2]]]]
    it "Y8" $ do
      let img = fromDynImage $ ImageY8 $ newImage 1 3 (map (\i -> ((i, 0), (fromIntegral i))) [0 .. 2])
      (asValue img :: [[[[Word8]]]]) `shouldBe` [[[[0], [1], [2]]]]
    it "YF" $ do
      let img = fromDynImage $ ImageYF $ newImage 1 3 (map (\i -> ((i, 0), (fromIntegral i))) [0 .. 2])
      (asValue img :: [[[[Float]]]]) `shouldBe` [[[[0], [1], [2]]]]
    it "YA8" $ do
      let img = fromDynImage $ ImageYA8 $ newImage 1 3 (map (\i -> ((i, 0), PixelYA8 (fromIntegral i) 0)) [0 .. 2])
      (asValue img :: [[[[Word8]]]]) `shouldBe` [[[[0, 0], [1, 0], [2, 0]]]]
    it "YA16" $ do
      let img = fromDynImage $ ImageYA16 $ newImage 1 3 (map (\i -> ((i, 0), PixelYA16 (fromIntegral i) 0)) [0 .. 2])
      (asValue img :: [[[[Int32]]]]) `shouldBe` [[[[0, 0], [1, 0], [2, 0]]]]
    it "RGB8" $ do
      let img = fromDynImage $ ImageRGB8 $ newImage 1 3 (map (\i -> ((i, 0), PixelRGB8 (fromIntegral i) 0 0)) [0 .. 2])
      (asValue img :: [[[[Word8]]]]) `shouldBe` [[[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]]
    it "RGBF" $ do
      let img = fromDynImage $ ImageRGBF $ newImage 1 3 (map (\i -> ((i, 0), PixelRGBF (fromIntegral i) 0 0)) [0 .. 2])
      (asValue img :: [[[[Float]]]]) `shouldBe` [[[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]]
    it "RGBA8" $ do
      let img = fromDynImage $ ImageRGBA8 $ newImage 1 3 (map (\i -> ((i, 0), PixelRGBA8 (fromIntegral i) 0 0 0)) [0 .. 2])
      (asValue img :: [[[[Word8]]]]) `shouldBe` [[[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]]]
    it "YCbCr8" $ do
      let img = fromDynImage $ ImageYCbCr8 $ newImage 1 3 (map (\i -> ((i, 0), PixelYCbCr8 (fromIntegral i) 0 0)) [0 .. 2])
      (asValue img :: [[[[Word8]]]]) `shouldBe` [[[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]]
    it "CMYK8" $ do
      let img = fromDynImage $ ImageCMYK8 $ newImage 1 3 (map (\i -> ((i, 0), PixelCMYK8 (fromIntegral i) 0 0 0)) [0 .. 2])
      (asValue img :: [[[[Word8]]]]) `shouldBe` [[[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]]]
    it "CMYK16" $ do
      let img = fromDynImage $ ImageCMYK16 $ newImage 1 3 (map (\i -> ((i, 0), PixelCMYK16 (fromIntegral i) 0 0 0)) [0 .. 2])
      (asValue img :: [[[[Int32]]]]) `shouldBe` [[[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]]]
  describe "fromImages" $ do
    it "RGB8" $ do
      let img = newImage 1 3 (map (\i -> ((i, 0), PixelRGB8 (fromIntegral i) 0 0)) [0 .. 2])
      imgs <- fromImages [img, img, img]
      (asValue imgs :: [[[[Word8]]]]) `shouldBe` [[[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]]
