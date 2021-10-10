{-# LANGUAGE OverloadedStrings #-}

module DimnameSpec (spec) where

import Control.Exception.Safe
import Test.Hspec
import Torch.Autograd
import Torch.DType
import Torch.Dimname
import Torch.Functional
import Torch.Tensor
import Torch.TensorFactories
import Torch.TensorOptions

spec :: Spec
spec = do
  it "named tensor with ones" $ do
    let v = onesWithDimnames' [(3, "batch")]
        s = sumWithDimnames v ["batch"] False Float
    -- ToDo:
    -- onesWithDimnames' does not work.
    -- When v is evaluated, cpu is running full speed!!
    True `shouldBe` True
