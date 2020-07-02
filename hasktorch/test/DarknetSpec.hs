{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ExtendedDefaultRules #-}

module DarknetSpec(spec) where

import Test.Hspec
import Control.Exception.Safe
import Control.Monad.State.Strict

import Torch.Tensor
import Torch.TensorFactories
import Torch.NN
import Torch.Typed.NN (HasForward(..))
import GHC.Exts
import Torch.Vision.Darknet.Config
import Torch.Vision.Darknet.Spec
import Torch.Vision.Darknet.Forward
import qualified Torch.Functional.Internal as I
import GHC.Generics

spec :: Spec
spec = do
  describe "index accesses" $ do
    it "index" $ do
      let v = asTensor ([1,2,3,4]::[Float])
          r = asValue (v @@ 2) :: Float
      r `shouldBe` 3.0
    it "index" $ do
      let v = asTensor (replicate 3 [1,2,3,4] :: [[Float]])
          r = asValue (v @@ (Ellipsis, 0))
      r `shouldBe` [1.0::Float,1.0,1.0]
    it "indexPut" $ do
      let v = asTensor ([1,2,3,4]::[Float])
          r = asValue (v @= (1,5.0::Float))
      r `shouldBe` [1.0::Float,5.0,3.0,4.0]
    it "indexPut" $ do
      let v = asTensor ([1,2,3,4]::[Float])
          r = asValue (v @= (1,5.0::Float))
      r `shouldBe` [1.0::Float,5.0,3.0,4.0]
  describe "DarknetSpec" $ do
    it "Convolution" $ do
      let spec' = ConvolutionSpec {
                   input_filters = 3,
                   filters = 16,
                   layer_size = 3,
                   stride = 1,
                   activation = "leaky"
                 }
      layer <- sample spec'
      shape (forward layer (ones' [1,3,416,416])) `shouldBe` [1,16,416,416]
    it "ConvolutionWithBatchNorm" $ do
      let spec' = ConvolutionWithBatchNormSpec {
                   input_filters = 3,
                   filters = 16,
                   layer_size = 3,
                   stride = 1,
                   activation = "leaky"
                 }
      layer <- sample spec'
      shape (forward layer (True,ones' [1,3,416,416])) `shouldBe` [1,16,416,416]
      shape (forward layer (False,ones' [1,3,416,416])) `shouldBe` [1,16,416,416]
    it "Read config" $ do
      mconfig <- readIniFile "test/yolov3-tiny.cfg"
      case mconfig of
        Right cfg@(DarknetConfig global layers) -> do
          length (toList layers) `shouldBe` 24
          case toDarknetSpec cfg of
            Right spec -> do
              length (show spec) > 0 `shouldBe` True
            Left err -> throwIO $ userError err
        Left err -> throwIO $ userError err
    it "Yolo:prediction" $ do
      let yolo = Torch.Vision.Darknet.Forward.Yolo [(23, 27), (37, 58), (81, 82)] 80 418
          pred = toPrediction yolo (ones' [1,255,10,10])
      shape (fromPrediction pred) `shouldBe` [1,3,10,10,85]
      shape (toX pred) `shouldBe` [1,3,10,10]
      shape (toPredClass pred) `shouldBe` [1,3,10,10,80]
    it "Yolo:grid" $ do
      shape (gridX 3) `shouldBe` [1,1,3,3]
      (asValue (gridX 3)::[[[[Float]]]]) `shouldBe` [[[[0.0,1.0,2.0],[0.0,1.0,2.0],[0.0,1.0,2.0]]]]
      (asValue (gridY 3)::[[[[Float]]]]) `shouldBe` [[[[0.0,0.0,0.0],[1.0,1.0,1.0],[2.0,2.0,2.0]]]]
      let scaled_anchors = toScaledAnchors [(81,82), (135,169)] (480.0/15.0)
      scaled_anchors `shouldBe` [(2.53125,2.5625),(4.21875,5.28125)]
      shape (toAnchorW scaled_anchors) `shouldBe` [1,2,1,1]
      (asValue (I.masked_select (asTensor ([1,2,3,4]::[Float]))  (asTensor ([True,False,True,False]::[Bool]))) :: [Float]) `shouldBe` [1.0,3.0]
      let v = zeros' [4]
      (asValue (I.index_put
               v
               ([asTensor ([2]::[Int])])
               (asTensor (12::Float))
               False)::[Float]) `shouldBe` [0.0,0.0,12.0,0.0]
      (asValue v ::[Float]) `shouldBe` [0.0,0.0,0.0,0.0]
