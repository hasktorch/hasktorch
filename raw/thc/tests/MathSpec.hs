module MathSpec (spec) where

import Foreign.C.Types
import Test.Hspec
-- import Torch.FFI.THC.Random

import qualified Torch.FFI.THC.Double.Tensor as D
import qualified Torch.FFI.THC.Double.TensorMath as D
import qualified Torch.FFI.THC.Double.TensorMathPointwise as D
import qualified Torch.FFI.THC.Double.TensorRandom as D


import Internal

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "something true" $ it "is true" $ True `shouldBe` True
  --describe "Math" $ withCudaState $ \g -> do
  --  it "Can initialize values with the fill method" $ do
  --    t1 <- D.c_newWithSize2d g 2 2
  --    D.c_fill  g t1 3.1
  --    r <- D.c_get2d g t1 0 0
  --    r `shouldBe` (3.1 :: CDouble)
  --    D.c_free  g t1
  --  it "Can invert double values with cinv" $ do
  --    t1 <- D.c_newWithSize2d g 3 2
  --    D.c_fill g t1 2.0
  --    result <- D.c_newWithSize2d g 3 2
  --    D.c_cinv g result t1
  --    r <- D.c_get2d g result 0 0
  --    r `shouldBe` (0.5 :: CDouble)
  --    r <- D.c_get2d g t1 0 0
  --    r `shouldBe` (2.0 :: CDouble)
  --    D.c_free g t1
  --    D.c_free g result

    -- cinv doesn't seem to be excluded by the preprocessor, yet is not implemented
    -- for Int
    -- it "Can invert int values with cinv (?)" $ do
    --   t1 <- c_THIntTensor_newWithSize2d 3 2
    --   c_THIntTensor_fill t1 2
    --   result <- c_THIntTensor_newWithSize2d 3 2
    --   c_THIntTensor_cinv result t1
    --   c_THIntTensor_get2d result 0 0 `shouldBe` (0 :: CInt)
    --   c_THIntTensor_get2d t1 0 0 `shouldBe` (2 :: CInt)
    --   c_THIntTensor_free t1
    --   c_THIntTensor_free result

