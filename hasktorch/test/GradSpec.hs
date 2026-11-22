module GradSpec (spec) where

import Control.Exception.Safe
import Test.Hspec
import Torch.Autograd
import Torch.DType
import Torch.Functional
import Torch.Tensor
import Torch.TensorFactories
import Torch.TensorOptions
import qualified Torch.Internal.Managed.Type.Tensor as ATen

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
  it "grad with respect to an input the output does not depend on" $ do
    xi <- makeIndependent $ ones' []
    unused <- makeIndependent $ ones' [2]
    let x = toDependent xi
        y = x * x + 5 * x + 3
        gs = grad y [xi, unused]
    length gs `shouldBe` 2
    fmap toDouble (Prelude.take 1 gs) `shouldBe` [7.0]
    definedFlags <- mapM (\(Unsafe t) -> ATen.tensor_defined t) gs
    definedFlags `shouldBe` [1, 0]
