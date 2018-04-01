module Torch.Indef.Static.Tensor.Math.Blas where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Blas as Dynamic
import qualified Torch.Class.Tensor.Math.Blas.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Blas ()

blasOp
  :: (Dimensions4 d d' d'' d''')
  => (Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ())
  -> Tensor d -> HsReal -> Tensor d' -> HsReal -> Tensor d'' -> Tensor d''' -> IO ()
blasOp fn r a x b y z = fn (asDynamic r) a (asDynamic x) b (asDynamic y) (asDynamic z)


instance Class.TensorMathBlas Tensor where
  addmv_  = blasOp Dynamic.addmv_
  addmm_  = blasOp Dynamic.addmm_
  addr_   = blasOp Dynamic.addr_
  addbmm_ = blasOp Dynamic.addbmm_
  baddbmm_ = blasOp Dynamic.baddbmm_
  dot a b = Dynamic.dot (asDynamic a) (asDynamic b)


