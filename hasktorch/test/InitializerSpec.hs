module InitializerSpec where

import Test.Hspec
import Torch.Functional
import Torch.Initializers
import Torch.Tensor (asValue)
import Prelude hiding (abs, mean, var)

spec :: Spec
spec = do
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
