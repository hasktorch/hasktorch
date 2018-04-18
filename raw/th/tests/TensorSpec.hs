module TensorSpec (spec) where

import Foreign
import Foreign.C.Types

import Test.Hspec

import Torch.FFI.Tests

import Torch.Types.TH (C'THState)
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
  describe "Float Tensor creation and access methods"  $ signedSuite nullPtr floatBook
  describe "Double Tensor creation and access methods" $ signedSuite nullPtr doubleBook
  describe "Byte Tensor creation and access methods"   $ signedSuite nullPtr byteBook
  describe "Int Tensor creation and access methods"    $ signedSuite nullPtr intBook
  describe "Long Tensor creation and access methods"   $ signedSuite nullPtr longBook
  describe "Short Tensor creation and access methods"  $ signedSuite nullPtr shortBook

type CState = Ptr C'THState

longBook :: TestFunctions CState (Ptr L.CTensor) L.CReal L.CAccReal
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
  , _dot = Just L.c_dot
  , _abs = Just L.c_abs
  }

shortBook :: TestFunctions CState (Ptr S.CTensor) S.CReal S.CAccReal
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
  , _dot = Just S.c_dot
  , _abs = Just S.c_abs
  }

floatBook :: TestFunctions CState (Ptr F.CTensor) F.CReal F.CAccReal
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
  , _dot = Just F.c_dot
  , _abs = Just F.c_abs
  }

doubleBook :: TestFunctions CState (Ptr D.CTensor) D.CReal D.CAccReal
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
  , _dot = Just D.c_dot
  , _abs = Just D.c_abs
  }

byteBook :: TestFunctions CState (Ptr B.CTensor) B.CReal B.CAccReal
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
  , _dot = Just B.c_dot
  , _abs = Nothing
  }

intBook :: TestFunctions CState (Ptr I.CTensor) I.CReal I.CAccReal
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
  , _dot = Just I.c_dot
  , _abs = Just I.c_abs
  }

