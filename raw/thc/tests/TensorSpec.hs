module TensorSpec (spec) where

import Foreign
import Foreign.C.Types

import Test.Hspec

import Torch.FFI.Tests

import Torch.Types.THC (C'THCState)
import qualified Torch.FFI.THC.General as General

import qualified Torch.Types.THC.Byte as B
import qualified Torch.FFI.THC.Byte.Tensor as B
import qualified Torch.FFI.THC.Byte.TensorIndex as B
import qualified Torch.FFI.THC.Byte.TensorMasked as B
import qualified Torch.FFI.THC.Byte.TensorMathBlas as B
import qualified Torch.FFI.THC.Byte.TensorMathCompare as B
import qualified Torch.FFI.THC.Byte.TensorMathCompareT as B
import qualified Torch.FFI.THC.Byte.TensorMath as B
import qualified Torch.FFI.THC.Byte.TensorMathPairwise as B
import qualified Torch.FFI.THC.Byte.TensorMathPointwise as B
import qualified Torch.FFI.THC.Byte.TensorMathReduce as B
import qualified Torch.FFI.THC.Byte.TensorMathScan as B
import qualified Torch.FFI.THC.Byte.TensorMode as B
import qualified Torch.FFI.THC.Byte.TensorRandom as B
import qualified Torch.FFI.THC.Byte.TensorScatterGather as B
import qualified Torch.FFI.THC.Byte.TensorSort as B
import qualified Torch.FFI.THC.Byte.TensorTopK as B

-- import qualified Torch.Types.THC.Float as F
-- import qualified Torch.FFI.THC.Float.Tensor as F
-- import qualified Torch.FFI.THC.Float.TensorIndex as F
-- import qualified Torch.FFI.THC.Float.TensorMasked as F
-- import qualified Torch.FFI.THC.Float.TensorMathBlas as F
-- import qualified Torch.FFI.THC.Float.TensorMathCompare as F
-- import qualified Torch.FFI.THC.Float.TensorMathCompareT as F
-- import qualified Torch.FFI.THC.Float.TensorMath as F
-- import qualified Torch.FFI.THC.Float.TensorMathPairwise as F
-- import qualified Torch.FFI.THC.Float.TensorMathPointwise as F
-- import qualified Torch.FFI.THC.Float.TensorMathReduce as F
-- import qualified Torch.FFI.THC.Float.TensorMathScan as F
-- import qualified Torch.FFI.THC.Float.TensorMode as F
-- import qualified Torch.FFI.THC.Float.TensorRandom as F
-- import qualified Torch.FFI.THC.Float.TensorScatterGather as F
-- import qualified Torch.FFI.THC.Float.TensorSort as F
-- import qualified Torch.FFI.THC.Float.TensorTopK as F

import qualified Torch.Types.THC.Double as D
import qualified Torch.FFI.THC.Double.Tensor as D
import qualified Torch.FFI.THC.Double.TensorIndex as D
import qualified Torch.FFI.THC.Double.TensorMasked as D
import qualified Torch.FFI.THC.Double.TensorMathBlas as D
import qualified Torch.FFI.THC.Double.TensorMathCompare as D
import qualified Torch.FFI.THC.Double.TensorMathCompareT as D
import qualified Torch.FFI.THC.Double.TensorMath as D
import qualified Torch.FFI.THC.Double.TensorMathPairwise as D
import qualified Torch.FFI.THC.Double.TensorMathPointwise as D
import qualified Torch.FFI.THC.Double.TensorMathReduce as D
import qualified Torch.FFI.THC.Double.TensorMathScan as D
import qualified Torch.FFI.THC.Double.TensorMode as D
import qualified Torch.FFI.THC.Double.TensorRandom as D
import qualified Torch.FFI.THC.Double.TensorScatterGather as D
import qualified Torch.FFI.THC.Double.TensorSort as D
import qualified Torch.FFI.THC.Double.TensorTopK as D

import qualified Torch.Types.THC.Int as I
import qualified Torch.FFI.THC.Int.Tensor as I
import qualified Torch.FFI.THC.Int.TensorIndex as I
import qualified Torch.FFI.THC.Int.TensorMasked as I
import qualified Torch.FFI.THC.Int.TensorMathBlas as I
import qualified Torch.FFI.THC.Int.TensorMathCompare as I
import qualified Torch.FFI.THC.Int.TensorMathCompareT as I
import qualified Torch.FFI.THC.Int.TensorMath as I
import qualified Torch.FFI.THC.Int.TensorMathPairwise as I
import qualified Torch.FFI.THC.Int.TensorMathPointwise as I
import qualified Torch.FFI.THC.Int.TensorMathReduce as I
import qualified Torch.FFI.THC.Int.TensorMathScan as I
import qualified Torch.FFI.THC.Int.TensorMode as I
import qualified Torch.FFI.THC.Int.TensorRandom as I
import qualified Torch.FFI.THC.Int.TensorScatterGather as I
import qualified Torch.FFI.THC.Int.TensorSort as I
import qualified Torch.FFI.THC.Int.TensorTopK as I

import qualified Torch.Types.THC.Short as S
import qualified Torch.FFI.THC.Short.Tensor as S
import qualified Torch.FFI.THC.Short.TensorIndex as S
import qualified Torch.FFI.THC.Short.TensorMasked as S
import qualified Torch.FFI.THC.Short.TensorMathBlas as S
import qualified Torch.FFI.THC.Short.TensorMathCompare as S
import qualified Torch.FFI.THC.Short.TensorMathCompareT as S
import qualified Torch.FFI.THC.Short.TensorMath as S
import qualified Torch.FFI.THC.Short.TensorMathPairwise as S
import qualified Torch.FFI.THC.Short.TensorMathPointwise as S
import qualified Torch.FFI.THC.Short.TensorMathReduce as S
import qualified Torch.FFI.THC.Short.TensorMathScan as S
import qualified Torch.FFI.THC.Short.TensorMode as S
import qualified Torch.FFI.THC.Short.TensorRandom as S
import qualified Torch.FFI.THC.Short.TensorScatterGather as S
import qualified Torch.FFI.THC.Short.TensorSort as S
import qualified Torch.FFI.THC.Short.TensorTopK as S

import qualified Torch.Types.THC.Long as L
import qualified Torch.FFI.THC.Long.Tensor as L
import qualified Torch.FFI.THC.Long.TensorIndex as L
import qualified Torch.FFI.THC.Long.TensorMasked as L
import qualified Torch.FFI.THC.Long.TensorMathBlas as L
import qualified Torch.FFI.THC.Long.TensorMathCompare as L
import qualified Torch.FFI.THC.Long.TensorMathCompareT as L
import qualified Torch.FFI.THC.Long.TensorMath as L
import qualified Torch.FFI.THC.Long.TensorMathPairwise as L
import qualified Torch.FFI.THC.Long.TensorMathPointwise as L
import qualified Torch.FFI.THC.Long.TensorMathReduce as L
import qualified Torch.FFI.THC.Long.TensorMathScan as L
import qualified Torch.FFI.THC.Long.TensorMode as L
import qualified Torch.FFI.THC.Long.TensorRandom as L
import qualified Torch.FFI.THC.Long.TensorScatterGather as L
import qualified Torch.FFI.THC.Long.TensorSort as L
import qualified Torch.FFI.THC.Long.TensorTopK as L

import Internal


main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  -- describe "Float Tensor creation and access methods"  $ withCudaState (`signedSuite` floatBook)
  describe "Double Tensor creation and access methods" $ withCudaState (`signedSuite` doubleBook)
  describe "Byte Tensor creation and access methods"   $ withCudaState (`signedSuite` byteBook)
  describe "Int Tensor creation and access methods"    $ withCudaState (`signedSuite` intBook)
  describe "Long Tensor creation and access methods"   $ withCudaState (`signedSuite` longBook)
  describe "Short Tensor creation and access methods"  $ withCudaState (`signedSuite` shortBook)


longBook :: TestFunctions (Ptr C'THCState) (Ptr L.CTensor) L.CReal L.CAccReal
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

shortBook :: TestFunctions (Ptr C'THCState) (Ptr S.CTensor) S.CReal S.CAccReal
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

{-
floatBook :: TestFunctions (Ptr C'THCState) (Ptr F.CTensor) F.CReal F.CAccReal
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
-}

doubleBook :: TestFunctions (Ptr C'THCState) (Ptr D.CTensor) D.CReal D.CAccReal
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

byteBook :: TestFunctions (Ptr C'THCState) (Ptr B.CTensor) B.CReal B.CAccReal
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

intBook :: TestFunctions (Ptr C'THCState) (Ptr I.CTensor) I.CReal I.CAccReal
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

