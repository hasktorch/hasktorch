module GradSpec (spec) where

import Control.Exception.Safe
import Test.Hspec
import Torch.Autograd
import Torch.DType
import Torch.Functional
import Torch.Tensor
import Torch.TensorFactories
import Torch.TensorOptions

spec :: Spec
spec = do
  it "grad with ones" $ do
    xi <- makeIndependent $ ones' []
    let x = toDependent xi
        y = x * x + 5 * x + 3
    fmap toDouble (grad y [xi]) `shouldBe` [7.0]
  it "grad with ones" $ do
    xi <- makeIndependent $ ones' []
    yi <- makeIndependent $ ones' []
    let x = toDependent xi
        y = toDependent yi
        z = x * x * y
    fmap toDouble (grad z [xi]) `shouldBe` [2.0]
    fmap toDouble (grad z [yi]) `shouldBe` [1.0]
