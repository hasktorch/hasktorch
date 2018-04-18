module Torch.Indef.Static.Tensor.Math.Blas () where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Blas as Dynamic
import qualified Torch.Class.Tensor.Math.Blas.Static as Class

import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Math ()
import Torch.Indef.Dynamic.Tensor.Math.Blas ()

blasOp
  :: (Dimensions4 d d' d'' d''')
  => (Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ())
  -> Tensor d -> HsReal -> Tensor d' -> HsReal -> Tensor d'' -> Tensor d''' -> IO ()
blasOp fn r a x b y z = fn (asDynamic r) a (asDynamic x) b (asDynamic y) (asDynamic z)


instance Class.TensorMathBlas Tensor where
  _addmv  = blasOp Dynamic._addmv
  _addmm  = blasOp Dynamic._addmm
  _addr r a x b y z = Dynamic._addr (asDynamic r) a (asDynamic x) b (asDynamic y) (asDynamic z)
  _addbmm = blasOp Dynamic._addbmm
  _baddbmm = blasOp Dynamic._baddbmm
  dot a b = Dynamic.dot (asDynamic a) (asDynamic b)


