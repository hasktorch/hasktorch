module InitializerSpec where

import Test.Hspec
import Torch.Functional
import Torch.Initializers
import Torch.Tensor (asValue)
import Prelude hiding (abs, max, mean, var)

spec :: Spec
spec = do
  describe "calculateFan" $ do
    -- reference values from torch.nn.init._calculate_fan_in_and_fan_out
    it "linear weight [out, in]" $ do
      calculateFan [50, 500] `shouldBe` (500, 50)
    it "conv weight [out, in, k, k] uses kernel dims only" $ do
      calculateFan [10, 30, 5, 5] `shouldBe` (30 * 5 * 5, 10 * 5 * 5)
  describe "Check initializers" $ do
    it "kaiming uniform is 0-centered" $ do
      x <- kaimingUniform' [50, 500]
      (asValue (abs . mean $ x) :: Float) < 0.01 `shouldBe` True
    it "kaiming normal is 0-centered" $ do
      x <- kaimingNormal' [50, 500]
      (asValue (abs . mean $ x) :: Float) < 0.01 `shouldBe` True
    it "xavier uniform is 0-centered" $ do
      x <- xavierUniform' [50, 500]
      (asValue (abs . mean $ x) :: Float) < 0.01 `shouldBe` True
    it "xavier normal is 0-centered" $ do
      x <- xavierNormal' [50, 500]
      (asValue (abs . mean $ x) :: Float) < 0.01 `shouldBe` True
    it "kaiming uniform matches PyTorch scaling on a conv shape" $ do
      x <- kaimingUniform' [10, 30, 5, 5]
      -- gain sqrt 2 for LeakyRelu 0, fan_in 750
      let bound = Prelude.sqrt 3.0 * Prelude.sqrt (2.0 / 750.0) :: Float
      (asValue (max . abs $ x) :: Float) <= bound + 1e-6 `shouldBe` True
      Prelude.abs ((asValue (std x) :: Float) - bound / Prelude.sqrt 3.0) < 0.005 `shouldBe` True
