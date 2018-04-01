module Torch.Indef.Dynamic.Tensor.Math.Blas where

import Foreign
import GHC.Int
import qualified Torch.Class.Tensor.Math.Blas as Class
import qualified Torch.Sig.Tensor.Math.Blas as Sig

import Torch.Indef.Types

blasOp
  :: (Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> CReal -> Ptr CTensor -> Ptr CTensor -> IO ())
  -> Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
blasOp fn r a x b y z =
  with2DynamicState r x $ \s' r' x' ->
    with2DynamicState y z $ \_ y' z' ->
      fn s' r' (hs2cReal a) x' (hs2cReal b) y' z'


instance Class.TensorMathBlas Dynamic where
  addmv_   = blasOp Sig.c_addmv
  addmm_   = blasOp Sig.c_addmm
  addr_    = blasOp Sig.c_addr
  addbmm_  = blasOp Sig.c_addbmm
  baddbmm_ = blasOp Sig.c_baddbmm

  dot :: Dynamic -> Dynamic -> IO HsAccReal
  dot a b = with2DynamicState a b $ fmap c2hsAccReal ..: Sig.c_dot



