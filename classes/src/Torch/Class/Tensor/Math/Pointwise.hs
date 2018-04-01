module Torch.Class.Tensor.Math.Pointwise where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Torch.Class.Types
import Torch.Dimensions

class TensorMathPointwise t where
  sign_        :: t -> t -> IO ()
  cross_       :: t -> t -> t -> DimVal -> IO ()
  clamp_       :: t -> t -> HsReal t -> HsReal t -> IO ()
  cadd_        :: t -> t -> HsReal t -> t -> IO ()
  csub_        :: t -> t -> HsReal t -> t -> IO ()
  cmul_        :: t -> t -> t -> IO ()
  cpow_        :: t -> t -> t -> IO ()
  cdiv_        :: t -> t -> t -> IO ()
  clshift_     :: t -> t -> t -> IO ()
  crshift_     :: t -> t -> t -> IO ()
  cfmod_       :: t -> t -> t -> IO ()
  cremainder_  :: t -> t -> t -> IO ()
  cmax_        :: t -> t -> t -> IO ()
  cmin_        :: t -> t -> t -> IO ()
  cmaxValue_   :: t -> t -> HsReal t -> IO ()
  cminValue_   :: t -> t -> HsReal t -> IO ()
  cbitand_     :: t -> t -> t -> IO ()
  cbitor_      :: t -> t -> t -> IO ()
  cbitxor_     :: t -> t -> t -> IO ()
  addcmul_     :: t -> t -> HsReal t -> t -> t -> IO ()
  addcdiv_     :: t -> t -> HsReal t -> t -> t -> IO ()

class TensorMathPointwiseSigned t where
  neg_ :: t -> t -> IO ()
  abs_ :: t -> t -> IO ()

class TensorMathPointwiseFloating t where
  cinv_         :: t -> t -> IO ()
  sigmoid_      :: t -> t -> IO ()
  log_          :: t -> t -> IO ()
  lgamma_       :: t -> t -> IO ()
  log1p_        :: t -> t -> IO ()
  exp_          :: t -> t -> IO ()
  cos_          :: t -> t -> IO ()
  acos_         :: t -> t -> IO ()
  cosh_         :: t -> t -> IO ()
  sin_          :: t -> t -> IO ()
  asin_         :: t -> t -> IO ()
  sinh_         :: t -> t -> IO ()
  tan_          :: t -> t -> IO ()
  atan_         :: t -> t -> IO ()
  atan2_        :: t -> t -> t -> IO ()
  tanh_         :: t -> t -> IO ()
  erf_          :: t -> t -> IO ()
  erfinv_       :: t -> t -> IO ()
  pow_          :: t -> t -> HsReal t -> IO ()
  tpow_         :: t -> HsReal t -> t -> IO ()
  sqrt_         :: t -> t -> IO ()
  rsqrt_        :: t -> t -> IO ()
  ceil_         :: t -> t -> IO ()
  floor_        :: t -> t -> IO ()
  round_        :: t -> t -> IO ()
  trunc_        :: t -> t -> IO ()
  frac_         :: t -> t -> IO ()
  lerp_         :: t -> t -> t -> HsReal t -> IO ()

class CPUTensorMathPointwiseFloating t where
  histc_        :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  bhistc_       :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()


