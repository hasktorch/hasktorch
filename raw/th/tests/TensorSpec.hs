module TensorSpec (spec) where

import Foreign.C.Types

import Test.Hspec

import qualified Torch.FFI.TH.Byte.Tensor as B
import qualified Torch.FFI.TH.Byte.TensorMath as B

import qualified Torch.FFI.TH.Float.Tensor as F
import qualified Torch.FFI.TH.Float.TensorMath as F

import qualified Torch.FFI.TH.Double.Tensor as D
import qualified Torch.FFI.TH.Double.TensorMath as D

import qualified Torch.FFI.TH.Int.Tensor as I
import qualified Torch.FFI.TH.Int.TensorMath as I

import qualified Torch.FFI.TH.Short.Tensor as S
import qualified Torch.FFI.TH.Short.TensorMath as S

import qualified Torch.FFI.TH.Long.Tensor as L
import qualified Torch.FFI.TH.Long.TensorMath as L

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "test floats"  testsFloat
  describe "test doubles" testsDouble
  describe "test bytes"   testsByte
  describe "test shorts"  testsShort
  describe "test ints"    testsInt
  describe "test longs"   testsLong

testsFloat :: Spec
testsFloat =
  describe "Float Tensor creation and access methods" $ do
    it "initializes empty tensor with 0 dimension" $ do
      t <- F.c_new ()
      F.c_nDimension () t >>= (`shouldBe` 0)
      F.c_free () t
    it "1D tensor has correct dimensions and sizes" $ do
      t <- F.c_newWithSize1d () 10
      F.c_nDimension () t >>= (`shouldBe` 1)
      F.c_size () t 0 >>= (`shouldBe` 10)
      F.c_free () t
    it "2D tensor has correct dimensions and sizes" $ do
      t <- F.c_newWithSize2d () 10 25
      F.c_nDimension () t >>= (`shouldBe` 2)
      F.c_size () t 0 >>= (`shouldBe` 10)
      F.c_size () t 1 >>= (`shouldBe` 25)
      F.c_free () t
    it "3D tensor has correct dimensions and sizes" $ do
      t <- F.c_newWithSize3d () 10 25 5
      F.c_nDimension () t >>= (`shouldBe` 3)
      F.c_size () t 0 >>= (`shouldBe` 10)
      F.c_size () t 1 >>= (`shouldBe` 25)
      F.c_size () t 2 >>= (`shouldBe` 5)
      F.c_free () t
    it "4D tensor has correct dimensions and sizes" $ do
      t <- F.c_newWithSize4d () 10 25 5 62
      F.c_nDimension () t >>= (`shouldBe` 4)
      F.c_size () t 0 >>= (`shouldBe` 10)
      F.c_size () t 1 >>= (`shouldBe` 25)
      F.c_size () t 2 >>= (`shouldBe` 5)
      F.c_size () t 3 >>= (`shouldBe` 62)
      F.c_free () t
    it "Can assign and retrieve correct 1D vector values" $ do
      t <- F.c_newWithSize1d () 10
      F.c_set1d () t 0 (CFloat 20.5)
      F.c_set1d () t 1 (CFloat 1.4)
      F.c_set1d () t 9 (CFloat 3.14)
      F.c_get1d () t 0 >>= (`shouldBe` (20.5 :: CFloat))
      F.c_get1d () t 1 >>= (`shouldBe` (1.4 :: CFloat))
      F.c_get1d () t 9 >>= (`shouldBe` (3.14 :: CFloat))
      F.c_free () t
    it "Can assign and retrieve correct 2D vector values" $ do
      t <- F.c_newWithSize2d () 10 15
      F.c_set2d () t 0 0 (CFloat 20.5)
      F.c_set2d () t 1 5 (CFloat 1.4)
      F.c_set2d () t 9 9 (CFloat 3.14)
      F.c_get2d () t 0 0 >>= (`shouldBe` (20.5 :: CFloat))
      F.c_get2d () t 1 5 >>= (`shouldBe` (1.4 :: CFloat))
      F.c_get2d () t 9 9 >>= (`shouldBe` (3.14 :: CFloat))
      F.c_free () t
    it "Can assign and retrieve correct 3D vector values" $ do
      t <- F.c_newWithSize3d () 10 15 10
      F.c_set3d () t 0 0 0 (CFloat 20.5)
      F.c_set3d () t 1 5 3 (CFloat 1.4)
      F.c_set3d () t 9 9 9 (CFloat 3.14)
      F.c_get3d () t 0 0 0 >>= (`shouldBe` (20.5 :: CFloat))
      F.c_get3d () t 1 5 3 >>= (`shouldBe` (1.4 :: CFloat))
      F.c_get3d () t 9 9 9 >>= (`shouldBe` (3.14 :: CFloat))
      F.c_free () t
    it "Can assign and retrieve correct 4D vector values" $ do
      t <- F.c_newWithSize4d () 10 15 10 20
      F.c_set4d () t 0 0 0 0 (CFloat 20.5)
      F.c_set4d () t 1 5 3 2 (CFloat 1.4)
      F.c_set4d () t 9 9 9 9 (CFloat 3.14)
      F.c_get4d () t 0 0 0 0 >>= (`shouldBe` (20.5 :: CFloat))
      F.c_get4d () t 1 5 3 2 >>= (`shouldBe` (1.4 :: CFloat))
      F.c_get4d () t 9 9 9 9 >>= (`shouldBe` (3.14 :: CFloat))
      F.c_free () t
    it "Can can initialize values with the fill method" $ do
      t1 <- F.c_newWithSize2d () 2 2
      F.c_fill () t1 3.1
      F.c_get2d () t1 0 0 >>= (`shouldBe` (3.1 :: CFloat))
      F.c_free () t1
    it "Can compute correct dot product between 1D vectors" $ do
      t1 <- F.c_newWithSize1d () 3
      t2 <- F.c_newWithSize1d () 3
      F.c_fill () t1 3.0
      F.c_fill () t2 4.0
      let value = F.c_dot () t1 t2
      value >>= (`shouldBe` 36.0)
      F.c_free () t1
      F.c_free () t2
    it "Can compute correct dot product between 2D tensors" $ do
      t1 <- F.c_newWithSize2d () 2 2
      t2 <- F.c_newWithSize2d () 2 2
      F.c_fill () t1 3.0
      F.c_fill () t2 4.0
      let value = F.c_dot () t1 t2
      value >>= (`shouldBe` 48.0)
      F.c_free () t1
      F.c_free () t2
    it "Can compute correct dot product between 3D tensors" $ do
      t1 <- F.c_newWithSize3d () 2 2 4
      t2 <- F.c_newWithSize3d () 2 2 4
      F.c_fill () t1 3.0
      F.c_fill () t2 4.0
      let value = F.c_dot () t1 t2
      value >>= (`shouldBe` 192.0)
      F.c_free () t1
      F.c_free () t2
    it "Can compute correct dot product between 4D tensors" $ do
      t1 <- F.c_newWithSize4d () 2 2 4 3
      t2 <- F.c_newWithSize4d () 2 2 4 3
      F.c_fill () t1 3.0
      F.c_fill () t2 4.0
      let value = F.c_dot () t1 t2
      value >>= (`shouldBe` 576.0)
      F.c_free () t1
      F.c_free () t2
    it "Can zero out values" $ do
      t1 <- F.c_newWithSize4d () 2 2 4 3
      F.c_fill () t1 3.0
      let value = F.c_dot () t1 t1
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- value >>= (`shouldBe` (432.0))
      F.c_zero () t1
      let value = F.c_dot () t1 t1
      value >>= (`shouldBe` 0.0)
      F.c_free () t1

    it "Can compute sum of all values" $ do
      t1 <- F.c_newWithSize3d () 2 2 4
      F.c_fill () t1 2.5
      F.c_sumall () t1 >>= (`shouldBe` 40.0)
      F.c_free () t1
    it "Can compute product of all values" $ do
      t1 <- F.c_newWithSize2d () 2 2
      F.c_fill () t1 1.5
      F.c_prodall () t1 >>= (`shouldBe` 5.0625)
      F.c_free () t1
    it "Can take abs of tensor values" $ do
      t1 <- F.c_newWithSize2d () 2 2
      F.c_fill () t1 (-1.5)
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- F.c_sumall () t1 >>= (`shouldBe` (-6.0))
      F.c_abs () t1 t1
      F.c_sumall () t1 >>= (`shouldBe` 6.0)
      F.c_free () t1

testsDouble :: Spec
testsDouble =
  describe "Double Tensor creation and access methods" $ do
    it "initializes empty tensor with 0 dimension" $ do
      t <- D.c_new ()
      D.c_nDimension () t >>= (`shouldBe` 0)
      D.c_free () t
    it "1D tensor has correct dimensions and sizes" $ do
      t <- D.c_newWithSize1d () 10
      D.c_nDimension () t >>= (`shouldBe` 1)
      D.c_size () t 0 >>= (`shouldBe` 10)
      D.c_free () t
    it "2D tensor has correct dimensions and sizes" $ do
      t <- D.c_newWithSize2d () 10 25
      D.c_nDimension () t >>= (`shouldBe` 2)
      D.c_size () t 0 >>= (`shouldBe` 10)
      D.c_size () t 1 >>= (`shouldBe` 25)
      D.c_free () t
    it "3D tensor has correct dimensions and sizes" $ do
      t <- D.c_newWithSize3d () 10 25 5
      D.c_nDimension () t >>= (`shouldBe` 3)
      D.c_size () t 0 >>= (`shouldBe` 10)
      D.c_size () t 1 >>= (`shouldBe` 25)
      D.c_size () t 2 >>= (`shouldBe` 5)
      D.c_free () t
    it "4D tensor has correct dimensions and sizes" $ do
      t <- D.c_newWithSize4d () 10 25 5 62
      D.c_nDimension () t >>= (`shouldBe` 4)
      D.c_size () t 0 >>= (`shouldBe` 10)
      D.c_size () t 1 >>= (`shouldBe` 25)
      D.c_size () t 2 >>= (`shouldBe` 5)
      D.c_size () t 3 >>= (`shouldBe` 62)
      D.c_free () t
    it "Can assign and retrieve correct 1D vector values" $ do
      t <- D.c_newWithSize1d () 10
      D.c_set1d () t 0 (CDouble 20.5)
      D.c_set1d () t 1 (CDouble 1.4)
      D.c_set1d () t 9 (CDouble 3.14)
      D.c_get1d () t 0 >>= (`shouldBe` (20.5 :: CDouble))
      D.c_get1d () t 1 >>= (`shouldBe` (1.4 :: CDouble))
      D.c_get1d () t 9 >>= (`shouldBe` (3.14 :: CDouble))
      D.c_free () t
    it "Can assign and retrieve correct 2D vector values" $ do
      t <- D.c_newWithSize2d () 10 15
      D.c_set2d () t 0 0 (CDouble 20.5)
      D.c_set2d () t 1 5 (CDouble 1.4)
      D.c_set2d () t 9 9 (CDouble 3.14)
      D.c_get2d () t 0 0 >>= (`shouldBe` (20.5 :: CDouble))
      D.c_get2d () t 1 5 >>= (`shouldBe` (1.4 :: CDouble))
      D.c_get2d () t 9 9 >>= (`shouldBe` (3.14 :: CDouble))
      D.c_free () t
    it "Can assign and retrieve correct 3D vector values" $ do
      t <- D.c_newWithSize3d () 10 15 10
      D.c_set3d () t 0 0 0 (CDouble 20.5)
      D.c_set3d () t 1 5 3 (CDouble 1.4)
      D.c_set3d () t 9 9 9 (CDouble 3.14)
      D.c_get3d () t 0 0 0 >>= (`shouldBe` (20.5 :: CDouble))
      D.c_get3d () t 1 5 3 >>= (`shouldBe` (1.4 :: CDouble))
      D.c_get3d () t 9 9 9 >>= (`shouldBe` (3.14 :: CDouble))
      D.c_free () t
    it "Can assign and retrieve correct 4D vector values" $ do
      t <- D.c_newWithSize4d () 10 15 10 20
      D.c_set4d () t 0 0 0 0 (CDouble 20.5)
      D.c_set4d () t 1 5 3 2 (CDouble 1.4)
      D.c_set4d () t 9 9 9 9 (CDouble 3.14)
      D.c_get4d () t 0 0 0 0 >>= (`shouldBe` (20.5 :: CDouble))
      D.c_get4d () t 1 5 3 2 >>= (`shouldBe` (1.4 :: CDouble))
      D.c_get4d () t 9 9 9 9 >>= (`shouldBe` (3.14 :: CDouble))
      D.c_free () t
    it "Can initialize values with the fill method" $ do
      t1 <- D.c_newWithSize2d () 2 2
      D.c_fill () t1 3.1
      D.c_get2d () t1 0 0 >>= (`shouldBe` (3.1 :: CDouble))
      D.c_free () t1
    it "Can compute correct dot product between 1D vectors" $ do
      t1 <- D.c_newWithSize1d () 3
      t2 <- D.c_newWithSize1d () 3
      D.c_fill () t1 3.0
      D.c_fill () t2 4.0
      let value = D.c_dot () t1 t2
      value >>= (`shouldBe` (36.0 :: CDouble))
      D.c_free () t1
      D.c_free () t2
    it "Can compute correct dot product between 2D tensors" $ do
      t1 <- D.c_newWithSize2d () 2 2
      t2 <- D.c_newWithSize2d () 2 2
      D.c_fill () t1 3.0
      D.c_fill () t2 4.0
      let value = D.c_dot () t1 t2
      value >>= (`shouldBe` (48.0 :: CDouble))
      D.c_free () t1
      D.c_free () t2
    it "Can compute correct dot product between 3D tensors" $ do
      t1 <- D.c_newWithSize3d () 2 2 4
      t2 <- D.c_newWithSize3d () 2 2 4
      D.c_fill () t1 3.0
      D.c_fill () t2 4.0
      let value = D.c_dot () t1 t2
      value >>= (`shouldBe` (192.0 :: CDouble))
      D.c_free () t1
      D.c_free () t2
    it "Can compute correct dot product between 4D tensors" $ do
      t1 <- D.c_newWithSize4d () 2 2 4 3
      t2 <- D.c_newWithSize4d () 2 2 4 3
      D.c_fill () t1 3.0
      D.c_fill () t2 4.0
      let value = D.c_dot () t1 t2
      value >>= (`shouldBe` (576.0 :: CDouble))
      D.c_free () t1
      D.c_free () t2
    it "Can zero out values" $ do
      t1 <- D.c_newWithSize4d () 2 2 4 3
      D.c_fill () t1 3.0
      let value = D.c_dot () t1 t1
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- value >>= (`shouldBe` (432.0 :: CDouble))
      D.c_zero () t1
      let value = D.c_dot () t1 t1
      value >>= (`shouldBe` (0.0 :: CDouble))
      D.c_free () t1
    it "Can compute sum of all values" $ do
      t1 <- D.c_newWithSize3d () 2 2 4
      D.c_fill () t1 2.5
      D.c_sumall () t1 >>= (`shouldBe` 40.0)
      D.c_free () t1
    it "Can compute product of all values" $ do
      t1 <- D.c_newWithSize2d () 2 2
      D.c_fill () t1 1.5
      D.c_prodall () t1 >>= (`shouldBe` 5.0625)
      D.c_free () t1
    it "Can take abs of tensor values" $ do
      t1 <- D.c_newWithSize2d () 2 2
      D.c_fill () t1 (-1.5)
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- D.c_sumall () t1 >>= (`shouldBe` (-6.0))
      D.c_abs () t1 t1
      D.c_sumall () t1 >>= (`shouldBe` (6.0))
      D.c_free () t1

testsInt :: Spec
testsInt =
  describe "Int Tensor creation and access methods" $ do
    it "initializes empty tensor with 0 dimension" $ do
      t <- I.c_new ()
      I.c_nDimension () t >>= (`shouldBe` 0)
      I.c_free () t
    it "1D tensor has correct dimensions and sizes" $ do
      t <- I.c_newWithSize1d () 10
      I.c_nDimension () t >>= (`shouldBe` 1)
      I.c_size () t 0 >>= (`shouldBe` 10)
      I.c_free () t
    it "2D tensor has correct dimensions and sizes" $ do
      t <- I.c_newWithSize2d () 10 25
      I.c_nDimension () t >>= (`shouldBe` 2)
      I.c_size () t 0 >>= (`shouldBe` 10)
      I.c_size () t 1 >>= (`shouldBe` 25)
      I.c_free () t
    it "3D tensor has correct dimensions and sizes" $ do
      t <- I.c_newWithSize3d () 10 25 5
      I.c_nDimension () t >>= (`shouldBe` 3)
      I.c_size () t 0 >>= (`shouldBe` 10)
      I.c_size () t 1 >>= (`shouldBe` 25)
      I.c_size () t 2 >>= (`shouldBe` 5)
      I.c_free () t
    it "4D tensor has correct dimensions and sizes" $ do
      t <- I.c_newWithSize4d () 10 25 5 62
      I.c_nDimension () t >>= (`shouldBe` 4)
      I.c_size () t 0 >>= (`shouldBe` 10)
      I.c_size () t 1 >>= (`shouldBe` 25)
      I.c_size () t 2 >>= (`shouldBe` 5)
      I.c_size () t 3 >>= (`shouldBe` 62)
      I.c_free () t
    it "Can assign and retrieve correct 1D vector values" $ do
      t <- I.c_newWithSize1d () 10
      I.c_set1d () t 0 (20)
      I.c_set1d () t 1 (1)
      I.c_set1d () t 9 (3)
      I.c_get1d () t 0 >>= (`shouldBe` (20))
      I.c_get1d () t 1 >>= (`shouldBe` (1))
      I.c_get1d () t 9 >>= (`shouldBe` (3))
      I.c_free () t
    it "Can assign and retrieve correct 2D vector values" $ do
      t <- I.c_newWithSize2d () 10 15
      I.c_set2d () t 0 0 (20)
      I.c_set2d () t 1 5 (1)
      I.c_set2d () t 9 9 (3)
      I.c_get2d () t 0 0 >>= (`shouldBe` (20))
      I.c_get2d () t 1 5 >>= (`shouldBe` (1))
      I.c_get2d () t 9 9 >>= (`shouldBe` (3))
      I.c_free () t
    it "Can assign and retrieve correct 3D vector values" $ do
      t <- I.c_newWithSize3d () 10 15 10
      I.c_set3d () t 0 0 0 (20)
      I.c_set3d () t 1 5 3 (1)
      I.c_set3d () t 9 9 9 (3)
      I.c_get3d () t 0 0 0 >>= (`shouldBe` (20))
      I.c_get3d () t 1 5 3 >>= (`shouldBe` (1))
      I.c_get3d () t 9 9 9 >>= (`shouldBe` (3))
      I.c_free () t
    it "Can assign and retrieve correct 4D vector values" $ do
      t <- I.c_newWithSize4d () 10 15 10 20
      I.c_set4d () t 0 0 0 0 (20)
      I.c_set4d () t 1 5 3 2 (1)
      I.c_set4d () t 9 9 9 9 (3)
      I.c_get4d () t 0 0 0 0 >>= (`shouldBe` (20))
      I.c_get4d () t 1 5 3 2 >>= (`shouldBe` (1))
      I.c_get4d () t 9 9 9 9 >>= (`shouldBe` (3))
      I.c_free () t
    it "Can can initialize values with the fill method" $ do
      t1 <- I.c_newWithSize2d () 2 2
      I.c_fill () t1 3
      I.c_get2d () t1 0 0 >>= (`shouldBe` (3))
      I.c_free () t1
    it "Can compute correct dot product between 1D vectors" $ do
      t1 <- I.c_newWithSize1d () 3
      t2 <- I.c_newWithSize1d () 3
      I.c_fill () t1 3
      I.c_fill () t2 4
      let value = I.c_dot () t1 t2
      value >>= (`shouldBe` 36)
      I.c_free () t1
      I.c_free () t2
    it "Can compute correct dot product between 2D tensors" $ do
      t1 <- I.c_newWithSize2d () 2 2
      t2 <- I.c_newWithSize2d () 2 2
      I.c_fill () t1 3
      I.c_fill () t2 4
      let value = I.c_dot () t1 t2
      value >>= (`shouldBe` 48)
      I.c_free () t1
      I.c_free () t2
    it "Can compute correct dot product between 3D tensors" $ do
      t1 <- I.c_newWithSize3d () 2 2 4
      t2 <- I.c_newWithSize3d () 2 2 4
      I.c_fill () t1 3
      I.c_fill () t2 4
      let value = I.c_dot () t1 t2
      value >>= (`shouldBe` 192)
      I.c_free () t1
      I.c_free () t2
    it "Can compute correct dot product between 4D tensors" $ do
      t1 <- I.c_newWithSize4d () 2 2 4 3
      t2 <- I.c_newWithSize4d () 2 2 4 3
      I.c_fill () t1 3
      I.c_fill () t2 4
      let value = I.c_dot () t1 t2
      value >>= (`shouldBe` 576)
      I.c_free () t1
      I.c_free () t2
    it "Can zero out values" $ do
      t1 <- I.c_newWithSize4d () 2 2 4 3
      I.c_fill () t1 3
      let value = I.c_dot () t1 t1
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- value >>= (`shouldBe` (432.0))
      I.c_zero () t1
      let value = I.c_dot () t1 t1
      value >>= (`shouldBe` 0)
      I.c_free () t1
    it "Can compute sum of all values" $ do
      t1 <- I.c_newWithSize3d () 2 2 4
      I.c_fill () t1 2
      I.c_sumall () t1 >>= (`shouldBe` 32)
      I.c_free () t1
    it "Can compute product of all values" $ do
      t1 <- I.c_newWithSize2d () 2 2
      I.c_fill () t1 2
      I.c_prodall () t1 >>= (`shouldBe` 16)
      I.c_free () t1
    it "Can take abs of tensor values" $ do
      t1 <- I.c_newWithSize2d () 2 2
      I.c_fill () t1 (-2)
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- I.c_sumall () t1 >>= (`shouldBe` (-6.0))
      I.c_abs () t1 t1
      I.c_sumall () t1 >>= (`shouldBe` (8))
      I.c_free () t1

testsByte :: Spec
testsByte = do
  describe "Byte Tensor creation and access methods" $ do
    it "initializes empty tensor with 0 dimension" $ do
      t <- B.c_new ()
      B.c_nDimension () t >>= (`shouldBe` 0)
      B.c_free () t
    it "1D tensor has correct dimensions and sizes" $ do
      t <- B.c_newWithSize1d () 10
      B.c_nDimension () t >>= (`shouldBe` 1)
      B.c_size () t 0 >>= (`shouldBe` 10)
      B.c_free () t
    it "2D tensor has correct dimensions and sizes" $ do
      t <- B.c_newWithSize2d () 10 25
      B.c_nDimension () t >>= (`shouldBe` 2)
      B.c_size () t 0 >>= (`shouldBe` 10)
      B.c_size () t 1 >>= (`shouldBe` 25)
      B.c_free () t
    it "3D tensor has correct dimensions and sizes" $ do
      t <- B.c_newWithSize3d () 10 25 5
      B.c_nDimension () t >>= (`shouldBe` 3)
      B.c_size () t 0 >>= (`shouldBe` 10)
      B.c_size () t 1 >>= (`shouldBe` 25)
      B.c_size () t 2 >>= (`shouldBe` 5)
      B.c_free () t
    it "4D tensor has correct dimensions and sizes" $ do
      t <- B.c_newWithSize4d () 10 25 5 62
      B.c_nDimension () t >>= (`shouldBe` 4)
      B.c_size () t 0 >>= (`shouldBe` 10)
      B.c_size () t 1 >>= (`shouldBe` 25)
      B.c_size () t 2 >>= (`shouldBe` 5)
      B.c_size () t 3 >>= (`shouldBe` 62)
      B.c_free () t
    it "Can assign and retrieve correct 1D vector values" $ do
      t <- B.c_newWithSize1d () 10
      B.c_set1d () t 0 (20)
      B.c_set1d () t 1 (1)
      B.c_set1d () t 9 (3)
      B.c_get1d () t 0 >>= (`shouldBe` (20))
      B.c_get1d () t 1 >>= (`shouldBe` (1))
      B.c_get1d () t 9 >>= (`shouldBe` (3))
      B.c_free () t
    it "Can assign and retrieve correct 2D vector values" $ do
      t <- B.c_newWithSize2d () 10 15
      B.c_set2d () t 0 0 (20)
      B.c_set2d () t 1 5 (1)
      B.c_set2d () t 9 9 (3)
      B.c_get2d () t 0 0 >>= (`shouldBe` (20))
      B.c_get2d () t 1 5 >>= (`shouldBe` (1))
      B.c_get2d () t 9 9 >>= (`shouldBe` (3))
      B.c_free () t
    it "Can assign and retrieve correct 3D vector values" $ do
      t <- B.c_newWithSize3d () 10 15 10
      B.c_set3d () t 0 0 0 (20)
      B.c_set3d () t 1 5 3 (1)
      B.c_set3d () t 9 9 9 (3)
      B.c_get3d () t 0 0 0 >>= (`shouldBe` (20))
      B.c_get3d () t 1 5 3 >>= (`shouldBe` (1))
      B.c_get3d () t 9 9 9 >>= (`shouldBe` (3))
      B.c_free () t
    it "Can assign and retrieve correct 4D vector values" $ do
      t <- B.c_newWithSize4d () 10 15 10 20
      B.c_set4d () t 0 0 0 0 (20)
      B.c_set4d () t 1 5 3 2 (1)
      B.c_set4d () t 9 9 9 9 (3)
      B.c_get4d () t 0 0 0 0 >>= (`shouldBe` (20))
      B.c_get4d () t 1 5 3 2 >>= (`shouldBe` (1))
      B.c_get4d () t 9 9 9 9 >>= (`shouldBe` (3))
      B.c_free () t
    it "Can can initialize values with the fill method" $ do
      t1 <- B.c_newWithSize2d () 2 2
      B.c_fill () t1 3
      B.c_get2d () t1 0 0 >>= (`shouldBe` (3))
      B.c_free () t1
    it "Can compute correct dot product between 1D vectors" $ do
      t1 <- B.c_newWithSize1d () 3
      t2 <- B.c_newWithSize1d () 3
      B.c_fill () t1 3
      B.c_fill () t2 4
      let value = B.c_dot () t1 t2
      value >>= (`shouldBe` 36)
      B.c_free () t1
      B.c_free () t2
    it "Can compute correct dot product between 2D tensors" $ do
      t1 <- B.c_newWithSize2d () 2 2
      t2 <- B.c_newWithSize2d () 2 2
      B.c_fill () t1 3
      B.c_fill () t2 4
      let value = B.c_dot () t1 t2
      value >>= (`shouldBe` 48)
      B.c_free () t1
      B.c_free () t2
    it "Can compute correct dot product between 3D tensors" $ do
      t1 <- B.c_newWithSize3d () 2 2 4
      t2 <- B.c_newWithSize3d () 2 2 4
      B.c_fill () t1 3
      B.c_fill () t2 4
      let value = B.c_dot () t1 t2
      value >>= (`shouldBe` 192)
      B.c_free () t1
      B.c_free () t2
    it "Can compute correct dot product between 4D tensors" $ do
      t1 <- B.c_newWithSize4d () 2 2 2 1
      t2 <- B.c_newWithSize4d () 2 2 2 1
      B.c_fill () t1 3
      B.c_fill () t2 4
      let value = B.c_dot () t1 t2
      value >>= (`shouldBe` 96)
      B.c_free () t1
      B.c_free () t2
    it "Can zero out values" $ do
      t1 <- B.c_newWithSize4d () 2 2 4 3
      B.c_fill () t1 3
      let value = B.c_dot () t1 t1
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- value >>= (`shouldBe` (432.0))
      B.c_zero () t1
      let value = B.c_dot () t1 t1
      value >>= (`shouldBe` 0)
      B.c_free () t1
    it "Can compute sum of all values" $ do
      t1 <- B.c_newWithSize3d () 2 2 4
      B.c_fill () t1 2
      B.c_sumall () t1 >>= (`shouldBe` 32)
      B.c_free () t1
    it "Can compute product of all values" $ do
      t1 <- B.c_newWithSize2d () 2 2
      B.c_fill () t1 2
      B.c_prodall () t1 >>= (`shouldBe` 16)
      B.c_free () t1

testsShort :: Spec
testsShort = do
  describe "Short Tensor creation and access methods" $ do
    it "initializes empty tensor with 0 dimension" $ do
      t <- S.c_new ()
      S.c_nDimension () t >>= (`shouldBe` 0)
      S.c_free () t
    it "1D tensor has correct dimensions and sizes" $ do
      t <- S.c_newWithSize1d () 10
      S.c_nDimension () t >>= (`shouldBe` 1)
      S.c_size () t 0 >>= (`shouldBe` 10)
      S.c_free () t
    it "2D tensor has correct dimensions and sizes" $ do
      t <- S.c_newWithSize2d () 10 25
      S.c_nDimension () t >>= (`shouldBe` 2)
      S.c_size () t 0 >>= (`shouldBe` 10)
      S.c_size () t 1 >>= (`shouldBe` 25)
      S.c_free () t
    it "3D tensor has correct dimensions and sizes" $ do
      t <- S.c_newWithSize3d () 10 25 5
      S.c_nDimension () t >>= (`shouldBe` 3)
      S.c_size () t 0 >>= (`shouldBe` 10)
      S.c_size () t 1 >>= (`shouldBe` 25)
      S.c_size () t 2 >>= (`shouldBe` 5)
      S.c_free () t
    it "4D tensor has correct dimensions and sizes" $ do
      t <- S.c_newWithSize4d () 10 25 5 62
      S.c_nDimension () t >>= (`shouldBe` 4)
      S.c_size () t 0 >>= (`shouldBe` 10)
      S.c_size () t 1 >>= (`shouldBe` 25)
      S.c_size () t 2 >>= (`shouldBe` 5)
      S.c_size () t 3 >>= (`shouldBe` 62)
      S.c_free () t
    it "Can assign and retrieve correct 1D vector values" $ do
      t <- S.c_newWithSize1d () 10
      S.c_set1d () t 0 (20)
      S.c_set1d () t 1 (1)
      S.c_set1d () t 9 (3)
      S.c_get1d () t 0 >>= (`shouldBe` (20))
      S.c_get1d () t 1 >>= (`shouldBe` (1))
      S.c_get1d () t 9 >>= (`shouldBe` (3))
      S.c_free () t
    it "Can assign and retrieve correct 2D vector values" $ do
      t <- S.c_newWithSize2d () 10 15
      S.c_set2d () t 0 0 (20)
      S.c_set2d () t 1 5 (1)
      S.c_set2d () t 9 9 (3)
      S.c_get2d () t 0 0 >>= (`shouldBe` (20))
      S.c_get2d () t 1 5 >>= (`shouldBe` (1))
      S.c_get2d () t 9 9 >>= (`shouldBe` (3))
      S.c_free () t
    it "Can assign and retrieve correct 3D vector values" $ do
      t <- S.c_newWithSize3d () 10 15 10
      S.c_set3d () t 0 0 0 (20)
      S.c_set3d () t 1 5 3 (1)
      S.c_set3d () t 9 9 9 (3)
      S.c_get3d () t 0 0 0 >>= (`shouldBe` (20))
      S.c_get3d () t 1 5 3 >>= (`shouldBe` (1))
      S.c_get3d () t 9 9 9 >>= (`shouldBe` (3))
      S.c_free () t
    it "Can assign and retrieve correct 4D vector values" $ do
      t <- S.c_newWithSize4d () 10 15 10 20
      S.c_set4d () t 0 0 0 0 (20)
      S.c_set4d () t 1 5 3 2 (1)
      S.c_set4d () t 9 9 9 9 (3)
      S.c_get4d () t 0 0 0 0 >>= (`shouldBe` (20))
      S.c_get4d () t 1 5 3 2 >>= (`shouldBe` (1))
      S.c_get4d () t 9 9 9 9 >>= (`shouldBe` (3))
      S.c_free () t
    it "Can can initialize values with the fill method" $ do
      t1 <- S.c_newWithSize2d () 2 2
      S.c_fill () t1 3
      S.c_get2d () t1 0 0 >>= (`shouldBe` (3))
      S.c_free () t1
    it "Can compute correct dot product between 1D vectors" $ do
      t1 <- S.c_newWithSize1d () 3
      t2 <- S.c_newWithSize1d () 3
      S.c_fill () t1 3
      S.c_fill () t2 4
      let value = S.c_dot () t1 t2
      value >>= (`shouldBe` 36)
      S.c_free () t1
      S.c_free () t2
    it "Can compute correct dot product between 2D tensors" $ do
      t1 <- S.c_newWithSize2d () 2 2
      t2 <- S.c_newWithSize2d () 2 2
      S.c_fill () t1 3
      S.c_fill () t2 4
      let value = S.c_dot () t1 t2
      value >>= (`shouldBe` 48)
      S.c_free () t1
      S.c_free () t2
    it "Can compute correct dot product between 3D tensors" $ do
      t1 <- S.c_newWithSize3d () 2 2 4
      t2 <- S.c_newWithSize3d () 2 2 4
      S.c_fill () t1 3
      S.c_fill () t2 4
      let value = S.c_dot () t1 t2
      value >>= (`shouldBe` 192)
      S.c_free () t1
      S.c_free () t2
    it "Can compute correct dot product between 4D tensors" $ do
      t1 <- S.c_newWithSize4d () 2 2 2 1
      t2 <- S.c_newWithSize4d () 2 2 2 1
      S.c_fill () t1 3
      S.c_fill () t2 4
      let value = S.c_dot () t1 t2
      value >>= (`shouldBe` 96)
      S.c_free () t1
      S.c_free () t2
    it "Can zero out values" $ do
      t1 <- S.c_newWithSize4d () 2 2 4 3
      S.c_fill () t1 3
      let value = S.c_dot () t1 t1
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- value >>= (`shouldBe` (432.0))
      S.c_zero () t1
      let value = S.c_dot () t1 t1
      value >>= (`shouldBe` 0)
      S.c_free () t1
    it "Can compute sum of all values" $ do
      t1 <- S.c_newWithSize3d () 2 2 4
      S.c_fill () t1 2
      S.c_sumall () t1 >>= (`shouldBe` 32)
      S.c_free () t1
    it "Can compute product of all values" $ do
      t1 <- S.c_newWithSize2d () 2 2
      S.c_fill () t1 2
      S.c_prodall () t1 >>= (`shouldBe` 16)
      S.c_free () t1
    it "Can take abs of tensor values" $ do
      t1 <- S.c_newWithSize2d () 2 2
      S.c_fill () t1 (-2)
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- S.c_sumall () t1 >>= (`shouldBe` (-6.0))
      S.c_abs () t1 t1
      S.c_sumall () t1 >>= (`shouldBe` (8))
      S.c_free () t1




testsLong :: Spec
testsLong = do
  describe "Long Tensor creation and access methods" $ do
    it "initializes empty tensor with 0 dimension" $ do
      t <- L.c_new ()
      L.c_nDimension () t >>= (`shouldBe` 0)
      L.c_free () t
    it "1D tensor has correct dimensions and sizes" $ do
      t <- L.c_newWithSize1d () 10
      L.c_nDimension () t >>= (`shouldBe` 1)
      L.c_size () t 0 >>= (`shouldBe` 10)
      L.c_free () t
    it "2D tensor has correct dimensions and sizes" $ do
      t <- L.c_newWithSize2d () 10 25
      L.c_nDimension () t >>= (`shouldBe` 2)
      L.c_size () t 0 >>= (`shouldBe` 10)
      L.c_size () t 1 >>= (`shouldBe` 25)
      L.c_free () t
    it "3D tensor has correct dimensions and sizes" $ do
      t <- L.c_newWithSize3d () 10 25 5
      L.c_nDimension () t >>= (`shouldBe` 3)
      L.c_size () t 0 >>= (`shouldBe` 10)
      L.c_size () t 1 >>= (`shouldBe` 25)
      L.c_size () t 2 >>= (`shouldBe` 5)
      L.c_free () t
    it "4D tensor has correct dimensions and sizes" $ do
      t <- L.c_newWithSize4d () 10 25 5 62
      L.c_nDimension () t >>= (`shouldBe` 4)
      L.c_size () t 0 >>= (`shouldBe` 10)
      L.c_size () t 1 >>= (`shouldBe` 25)
      L.c_size () t 2 >>= (`shouldBe` 5)
      L.c_size () t 3 >>= (`shouldBe` 62)
      L.c_free () t
    it "Can assign and retrieve correct 1D vector values" $ do
      t <- L.c_newWithSize1d () 10
      L.c_set1d () t 0 (20)
      L.c_set1d () t 1 (1)
      L.c_set1d () t 9 (3)
      L.c_get1d () t 0 >>= (`shouldBe` (20))
      L.c_get1d () t 1 >>= (`shouldBe` (1))
      L.c_get1d () t 9 >>= (`shouldBe` (3))
      L.c_free () t
    it "Can assign and retrieve correct 2D vector values" $ do
      t <- L.c_newWithSize2d () 10 15
      L.c_set2d () t 0 0 (20)
      L.c_set2d () t 1 5 (1)
      L.c_set2d () t 9 9 (3)
      L.c_get2d () t 0 0 >>= (`shouldBe` (20))
      L.c_get2d () t 1 5 >>= (`shouldBe` (1))
      L.c_get2d () t 9 9 >>= (`shouldBe` (3))
      L.c_free () t
    it "Can assign and retrieve correct 3D vector values" $ do
      t <- L.c_newWithSize3d () 10 15 10
      L.c_set3d () t 0 0 0 (20)
      L.c_set3d () t 1 5 3 (1)
      L.c_set3d () t 9 9 9 (3)
      L.c_get3d () t 0 0 0 >>= (`shouldBe` (20))
      L.c_get3d () t 1 5 3 >>= (`shouldBe` (1))
      L.c_get3d () t 9 9 9 >>= (`shouldBe` (3))
      L.c_free () t
    it "Can assign and retrieve correct 4D vector values" $ do
      t <- L.c_newWithSize4d () 10 15 10 20
      L.c_set4d () t 0 0 0 0 (20)
      L.c_set4d () t 1 5 3 2 (1)
      L.c_set4d () t 9 9 9 9 (3)
      L.c_get4d () t 0 0 0 0 >>= (`shouldBe` (20))
      L.c_get4d () t 1 5 3 2 >>= (`shouldBe` (1))
      L.c_get4d () t 9 9 9 9 >>= (`shouldBe` (3))
      L.c_free () t
    it "Can can initialize values with the fill method" $ do
      t1 <- L.c_newWithSize2d () 2 2
      L.c_fill () t1 3
      L.c_get2d () t1 0 0 >>= (`shouldBe` (3))
      L.c_free () t1
    it "Can compute correct dot product between 1D vectors" $ do
      t1 <- L.c_newWithSize1d () 3
      t2 <- L.c_newWithSize1d () 3
      L.c_fill () t1 3
      L.c_fill () t2 4
      let value = L.c_dot () t1 t2
      value >>= (`shouldBe` 36)
      L.c_free () t1
      L.c_free () t2
    it "Can compute correct dot product between 2D tensors" $ do
      t1 <- L.c_newWithSize2d () 2 2
      t2 <- L.c_newWithSize2d () 2 2
      L.c_fill () t1 3
      L.c_fill () t2 4
      let value = L.c_dot () t1 t2
      value >>= (`shouldBe` 48)
      L.c_free () t1
      L.c_free () t2
    it "Can compute correct dot product between 3D tensors" $ do
      t1 <- L.c_newWithSize3d () 2 2 4
      t2 <- L.c_newWithSize3d () 2 2 4
      L.c_fill () t1 3
      L.c_fill () t2 4
      let value = L.c_dot () t1 t2
      value >>= (`shouldBe` 192)
      L.c_free () t1
      L.c_free () t2
    it "Can compute correct dot product between 4D tensors" $ do
      t1 <- L.c_newWithSize4d () 2 2 2 1
      t2 <- L.c_newWithSize4d () 2 2 2 1
      L.c_fill () t1 3
      L.c_fill () t2 4
      let value = L.c_dot () t1 t2
      value >>= (`shouldBe` 96)
      L.c_free () t1
      L.c_free () t2
    it "Can zero out values" $ do
      t1 <- L.c_newWithSize4d () 2 2 4 3
      L.c_fill () t1 3
      let value = L.c_dot () t1 t1
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- value >>= (`shouldBe` (432.0))
      L.c_zero () t1
      let value = L.c_dot () t1 t1
      value >>= (`shouldBe` 0)
      L.c_free () t1
    it "Can compute sum of all values" $ do
      t1 <- L.c_newWithSize3d () 2 2 4
      L.c_fill () t1 2
      L.c_sumall () t1 >>= (`shouldBe` 32)
      L.c_free () t1
    it "Can compute product of all values" $ do
      t1 <- L.c_newWithSize2d () 2 2
      L.c_fill () t1 2
      L.c_prodall () t1 >>= (`shouldBe` 16)
      L.c_free () t1
    it "Can take abs of tensor values" $ do
      t1 <- L.c_newWithSize2d () 2 2
      L.c_fill () t1 (-2)
      -- sequencing does not work if there is more than one shouldBe test in
      -- an "it" monad
      -- L.c_sumall () t1 >>= (`shouldBe` (-6.0))
      L.c_abs () t1 t1
      L.c_sumall () t1 >>= (`shouldBe` 8)
      L.c_free () t1

