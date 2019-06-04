module GradSpec (spec) where

import Test.Hspec
import Control.Exception.Safe

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

spec :: Spec
spec = do
  it "grad with ones" $ do
    xi <- makeIndependent $ ones' []
    let x = toDependent xi
        y = x * x + 5 * x + 3
    fmap asDouble (grad y [xi]) `shouldBe` [7.0]
  it "grad with ones" $ do
    xi <- makeIndependent $ ones' []
    yi <- makeIndependent $ ones' []
    let x = toDependent xi
        y = toDependent yi
        z = x * x * y
    fmap asDouble (grad z [xi]) `shouldBe` [2.0]
    fmap asDouble (grad z [yi]) `shouldBe` [1.0]
