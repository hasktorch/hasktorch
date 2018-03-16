{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
module TensorSpec (spec) where

import Foreign
import Foreign.C.Types

import Test.Hspec

import qualified Torch.Types.TH.Byte as B
import qualified Torch.FFI.TH.Byte.Tensor as B
import qualified Torch.FFI.TH.Byte.TensorMath as B

import qualified Torch.Types.TH.Float as F
import qualified Torch.FFI.TH.Float.Tensor as F
import qualified Torch.FFI.TH.Float.TensorMath as F

import qualified Torch.Types.TH.Double as D
import qualified Torch.FFI.TH.Double.Tensor as D
import qualified Torch.FFI.TH.Double.TensorMath as D

import qualified Torch.Types.TH.Int as I
import qualified Torch.FFI.TH.Int.Tensor as I
import qualified Torch.FFI.TH.Int.TensorMath as I

import qualified Torch.Types.TH.Short as S
import qualified Torch.FFI.TH.Short.Tensor as S
import qualified Torch.FFI.TH.Short.TensorMath as S

import qualified Torch.Types.TH.Long as L
import qualified Torch.FFI.TH.Long.Tensor as L
import qualified Torch.FFI.TH.Long.TensorMath as L

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Float Tensor creation and access methods"  $ signedSuite floatBook
  describe "Double Tensor creation and access methods" $ signedSuite doubleBook
  describe "Byte Tensor creation and access methods"   $ signedSuite byteBook
  describe "Int Tensor creation and access methods"    $ signedSuite intBook
  describe "Long Tensor creation and access methods"   $ signedSuite longBook
  describe "Short Tensor creation and access methods"  $ signedSuite shortBook

data TestFunctions state tensor real accreal = TestFunctions
  { _new :: state -> IO tensor
  , _newWithSize1d :: state -> CLLong -> IO tensor
  , _newWithSize2d :: state -> CLLong -> CLLong -> IO tensor
  , _newWithSize3d :: state -> CLLong -> CLLong -> CLLong -> IO tensor
  , _newWithSize4d :: state -> CLLong -> CLLong -> CLLong -> CLLong -> IO tensor
  , _nDimension :: state -> tensor -> IO CInt
  , _set1d :: state -> tensor -> CLLong -> real -> IO ()
  , _get1d :: state -> tensor -> CLLong -> IO real
  , _set2d :: state -> tensor -> CLLong -> CLLong -> real -> IO ()
  , _get2d :: state -> tensor -> CLLong -> CLLong -> IO real
  , _set3d :: state -> tensor -> CLLong -> CLLong -> CLLong -> real -> IO ()
  , _get3d :: state -> tensor -> CLLong -> CLLong -> CLLong -> IO real
  , _set4d :: state -> tensor -> CLLong -> CLLong -> CLLong -> CLLong -> real -> IO ()
  , _get4d :: state -> tensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO real
  , _size :: state -> tensor -> CInt -> IO CLLong
  , _fill :: state -> tensor -> real -> IO ()
  , _free :: state -> tensor -> IO ()
  , _sumall :: state -> tensor -> IO accreal
  , _prodall :: state -> tensor -> IO accreal
  , _zero :: state -> tensor -> IO ()
  , _dot :: state -> tensor -> tensor -> IO accreal
  , _abs :: Maybe (state -> tensor -> tensor -> IO ())
  }

longBook :: TestFunctions () (Ptr L.CTensor) L.CReal L.CAccReal
longBook = TestFunctions
  { _new = L.c_new
  , _newWithSize1d = L.c_newWithSize1d
  , _newWithSize2d = L.c_newWithSize2d
  , _newWithSize3d = L.c_newWithSize3d
  , _newWithSize4d = L.c_newWithSize4d
  , _nDimension = L.c_nDimension
  , _set1d = L.c_set1d
  , _get1d = L.c_get1d
  , _set2d = L.c_set2d
  , _get2d = L.c_get2d
  , _set3d = L.c_set3d
  , _get3d = L.c_get3d
  , _set4d = L.c_set4d
  , _get4d = L.c_get4d
  , _size = L.c_size
  , _fill = L.c_fill
  , _free = L.c_free
  , _sumall = L.c_sumall
  , _prodall = L.c_prodall
  , _zero = L.c_zero
  , _dot = L.c_dot
  , _abs = Just L.c_abs
  }

shortBook :: TestFunctions () (Ptr S.CTensor) S.CReal S.CAccReal
shortBook = TestFunctions
  { _new = S.c_new
  , _newWithSize1d = S.c_newWithSize1d
  , _newWithSize2d = S.c_newWithSize2d
  , _newWithSize3d = S.c_newWithSize3d
  , _newWithSize4d = S.c_newWithSize4d
  , _nDimension = S.c_nDimension
  , _set1d = S.c_set1d
  , _get1d = S.c_get1d
  , _set2d = S.c_set2d
  , _get2d = S.c_get2d
  , _set3d = S.c_set3d
  , _get3d = S.c_get3d
  , _set4d = S.c_set4d
  , _get4d = S.c_get4d
  , _size = S.c_size
  , _fill = S.c_fill
  , _free = S.c_free
  , _sumall = S.c_sumall
  , _prodall = S.c_prodall
  , _zero = S.c_zero
  , _dot = S.c_dot
  , _abs = Just S.c_abs
  }

floatBook :: TestFunctions () (Ptr F.CTensor) F.CReal F.CAccReal
floatBook = TestFunctions
  { _new = F.c_new
  , _newWithSize1d = F.c_newWithSize1d
  , _newWithSize2d = F.c_newWithSize2d
  , _newWithSize3d = F.c_newWithSize3d
  , _newWithSize4d = F.c_newWithSize4d
  , _nDimension = F.c_nDimension
  , _set1d = F.c_set1d
  , _get1d = F.c_get1d
  , _set2d = F.c_set2d
  , _get2d = F.c_get2d
  , _set3d = F.c_set3d
  , _get3d = F.c_get3d
  , _set4d = F.c_set4d
  , _get4d = F.c_get4d
  , _size = F.c_size
  , _fill = F.c_fill
  , _free = F.c_free
  , _sumall = F.c_sumall
  , _prodall = F.c_prodall
  , _zero = F.c_zero
  , _dot = F.c_dot
  , _abs = Just F.c_abs
  }

doubleBook :: TestFunctions () (Ptr D.CTensor) D.CReal D.CAccReal
doubleBook = TestFunctions
  { _new = D.c_new
  , _newWithSize1d = D.c_newWithSize1d
  , _newWithSize2d = D.c_newWithSize2d
  , _newWithSize3d = D.c_newWithSize3d
  , _newWithSize4d = D.c_newWithSize4d
  , _nDimension = D.c_nDimension
  , _set1d = D.c_set1d
  , _get1d = D.c_get1d
  , _set2d = D.c_set2d
  , _get2d = D.c_get2d
  , _set3d = D.c_set3d
  , _get3d = D.c_get3d
  , _set4d = D.c_set4d
  , _get4d = D.c_get4d
  , _size = D.c_size
  , _fill = D.c_fill
  , _free = D.c_free
  , _sumall = D.c_sumall
  , _prodall = D.c_prodall
  , _zero = D.c_zero
  , _dot = D.c_dot
  , _abs = Just D.c_abs
  }

byteBook :: TestFunctions () (Ptr B.CTensor) B.CReal B.CAccReal
byteBook = TestFunctions
  { _new = B.c_new
  , _newWithSize1d = B.c_newWithSize1d
  , _newWithSize2d = B.c_newWithSize2d
  , _newWithSize3d = B.c_newWithSize3d
  , _newWithSize4d = B.c_newWithSize4d
  , _nDimension = B.c_nDimension
  , _set1d = B.c_set1d
  , _get1d = B.c_get1d
  , _set2d = B.c_set2d
  , _get2d = B.c_get2d
  , _set3d = B.c_set3d
  , _get3d = B.c_get3d
  , _set4d = B.c_set4d
  , _get4d = B.c_get4d
  , _size = B.c_size
  , _fill = B.c_fill
  , _free = B.c_free
  , _sumall = B.c_sumall
  , _prodall = B.c_prodall
  , _zero = B.c_zero
  , _dot = B.c_dot
  , _abs = Nothing
  }

intBook :: TestFunctions () (Ptr I.CTensor) I.CReal I.CAccReal
intBook = TestFunctions
  { _new = I.c_new
  , _newWithSize1d = I.c_newWithSize1d
  , _newWithSize2d = I.c_newWithSize2d
  , _newWithSize3d = I.c_newWithSize3d
  , _newWithSize4d = I.c_newWithSize4d
  , _nDimension = I.c_nDimension
  , _set1d = I.c_set1d
  , _get1d = I.c_get1d
  , _set2d = I.c_set2d
  , _get2d = I.c_get2d
  , _set3d = I.c_set3d
  , _get3d = I.c_get3d
  , _set4d = I.c_set4d
  , _get4d = I.c_get4d
  , _size = I.c_size
  , _fill = I.c_fill
  , _free = I.c_free
  , _sumall = I.c_sumall
  , _prodall = I.c_prodall
  , _zero = I.c_zero
  , _dot = I.c_dot
  , _abs = Just I.c_abs
  }

type RealConstr n = (Num n, Show n, Eq n)

signedSuite :: (RealConstr real, RealConstr accreal) => TestFunctions () tensor real accreal -> Spec
signedSuite fs = do
  it "initializes empty tensor with 0 dimension" $ do
    t <- new ()
    nDimension () t >>= (`shouldBe` 0)
    free () t
  it "1D tensor has correct dimensions and sizes" $ do
    t <- newWithSize1d () 10
    nDimension () t >>= (`shouldBe` 1)
    size () t 0 >>= (`shouldBe` 10)
    free () t
  it "2D tensor has correct dimensions and sizes" $ do
    t <- newWithSize2d () 10 25
    nDimension () t >>= (`shouldBe` 2)
    size () t 0 >>= (`shouldBe` 10)
    size () t 1 >>= (`shouldBe` 25)
    free () t
  it "3D tensor has correct dimensions and sizes" $ do
    t <- newWithSize3d () 10 25 5
    nDimension () t >>= (`shouldBe` 3)
    size () t 0 >>= (`shouldBe` 10)
    size () t 1 >>= (`shouldBe` 25)
    size () t 2 >>= (`shouldBe` 5)
    free () t
  it "4D tensor has correct dimensions and sizes" $ do
    t <- newWithSize4d () 10 25 5 62
    nDimension () t >>= (`shouldBe` 4)
    size () t 0 >>= (`shouldBe` 10)
    size () t 1 >>= (`shouldBe` 25)
    size () t 2 >>= (`shouldBe` 5)
    size () t 3 >>= (`shouldBe` 62)
    free () t

  it "Can assign and retrieve correct 1D vector values" $ do
    t <- newWithSize1d () 10
    set1d () t 0 (20)
    set1d () t 1 (1)
    set1d () t 9 (3)
    get1d () t 0 >>= (`shouldBe` (20))
    get1d () t 1 >>= (`shouldBe` (1))
    get1d () t 9 >>= (`shouldBe` (3))
    free () t
  it "Can assign and retrieve correct 2D vector values" $ do
    t <- newWithSize2d () 10 15
    set2d () t 0 0 (20)
    set2d () t 1 5 (1)
    set2d () t 9 9 (3)
    get2d () t 0 0 >>= (`shouldBe` (20))
    get2d () t 1 5 >>= (`shouldBe` (1))
    get2d () t 9 9 >>= (`shouldBe` (3))
    free () t
  it "Can assign and retrieve correct 3D vector values" $ do
    t <- newWithSize3d () 10 15 10
    set3d () t 0 0 0 (20)
    set3d () t 1 5 3 (1)
    set3d () t 9 9 9 (3)
    get3d () t 0 0 0 >>= (`shouldBe` (20))
    get3d () t 1 5 3 >>= (`shouldBe` (1))
    get3d () t 9 9 9 >>= (`shouldBe` (3))
    free () t
  it "Can assign and retrieve correct 4D vector values" $ do
    t <- newWithSize4d () 10 15 10 20
    set4d () t 0 0 0 0 (20)
    set4d () t 1 5 3 2 (1)
    set4d () t 9 9 9 9 (3)
    get4d () t 0 0 0 0 >>= (`shouldBe` (20))
    get4d () t 1 5 3 2 >>= (`shouldBe` (1))
    get4d () t 9 9 9 9 >>= (`shouldBe` (3))
    free () t
  it "Can can initialize values with the fill method" $ do
    t1 <- newWithSize2d () 2 2
    fill () t1 3
    get2d () t1 0 0 >>= (`shouldBe` (3))
    free () t1
  it "Can compute correct dot product between 1D vectors" $ do
    t1 <- newWithSize1d () 3
    t2 <- newWithSize1d () 3
    fill () t1 3
    fill () t2 4
    let value = dot () t1 t2
    value >>= (`shouldBe` 36)
    free () t1
    free () t2
  it "Can compute correct dot product between 2D tensors" $ do
    t1 <- newWithSize2d () 2 2
    t2 <- newWithSize2d () 2 2
    fill () t1 3
    fill () t2 4
    let value = dot () t1 t2
    value >>= (`shouldBe` 48)
    free () t1
    free () t2
  it "Can compute correct dot product between 3D tensors" $ do
    t1 <- newWithSize3d () 2 2 4
    t2 <- newWithSize3d () 2 2 4
    fill () t1 3
    fill () t2 4
    let value = dot () t1 t2
    value >>= (`shouldBe` 192)
    free () t1
    free () t2
  it "Can compute correct dot product between 4D tensors" $ do
    t1 <- newWithSize4d () 2 2 2 1
    t2 <- newWithSize4d () 2 2 2 1
    fill () t1 3
    fill () t2 4
    let value = dot () t1 t2
    value >>= (`shouldBe` 96)
    free () t1
    free () t2
  it "Can zero out values" $ do
    t1 <- newWithSize4d () 2 2 4 3
    fill () t1 3
    -- let value = dot () t1 t1
    -- sequencing does not work if there is more than one shouldBe test in
    -- an "it" monad
    -- value >>= (`shouldBe` (432.0))
    zero () t1
    dot () t1 t1 >>= (`shouldBe` 0)
    free () t1
  it "Can compute sum of all values" $ do
    t1 <- newWithSize3d () 2 2 4
    fill () t1 2
    sumall () t1 >>= (`shouldBe` 32)
    free () t1
  it "Can compute product of all values" $ do
    t1 <- newWithSize2d () 2 2
    fill () t1 2
    prodall () t1 >>= (`shouldBe` 16)
    free () t1
  case mabs of
    Nothing  -> pure ()
    Just abs ->
      it "Can take abs of tensor values" $ do
        t1 <- newWithSize2d () 2 2
        fill () t1 (-2)
        -- sequencing does not work if there is more than one shouldBe test in
        -- an "it" monad
        -- sumall () t1 >>= (`shouldBe` (-6.0))
        abs () t1 t1
        sumall () t1 >>= (`shouldBe` 8)
        free () t1
 where
  new = _new fs
  newWithSize1d = _newWithSize1d fs
  newWithSize2d = _newWithSize2d fs
  newWithSize3d = _newWithSize3d fs
  newWithSize4d = _newWithSize4d fs
  nDimension = _nDimension fs
  set1d = _set1d fs
  get1d = _get1d fs
  set2d = _set2d fs
  get2d = _get2d fs
  set3d = _set3d fs
  get3d = _get3d fs
  set4d = _set4d fs
  get4d = _get4d fs
  size = _size fs
  fill = _fill fs
  free = _free fs
  sumall = _sumall fs
  mabs = _abs fs
  prodall = _prodall fs
  dot = _dot fs
  zero = _zero fs


