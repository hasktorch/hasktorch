module Torch.Class.Tensor.Math.Pointwise where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Torch.Class.Types

class TensorMathPointwise t where
  sign_        :: t -> t -> io ()
  cross_       :: t -> t -> t -> Int32 -> io ()
  clamp_       :: t -> t -> HsReal t -> HsReal t -> io ()
  cadd_        :: t -> t -> HsReal t -> t -> io ()
  csub_        :: t -> t -> HsReal t -> t -> io ()
  cmul_        :: t -> t -> t -> io ()
  cpow_        :: t -> t -> t -> io ()
  cdiv_        :: t -> t -> t -> io ()
  clshift_     :: t -> t -> t -> io ()
  crshift_     :: t -> t -> t -> io ()
  cfmod_       :: t -> t -> t -> io ()
  cremainder_  :: t -> t -> t -> io ()
  cmax_        :: t -> t -> t -> io ()
  cmin_        :: t -> t -> t -> io ()
  cmaxValue_   :: t -> t -> HsReal t -> io ()
  cminValue_   :: t -> t -> HsReal t -> io ()
  cbitand_     :: t -> t -> t -> io ()
  cbitor_      :: t -> t -> t -> io ()
  cbitxor_     :: t -> t -> t -> io ()
  addcmul_     :: t -> t -> HsReal t -> t -> t -> io ()
  addcdiv_     :: t -> t -> HsReal t -> t -> t -> io ()

class TensorMathPointwiseSigned t where
  neg_ :: t -> t -> IO ()
  abs_ :: t -> t -> IO ()

class TensorMathPointwiseFloating t where
  cinv_         :: t -> t -> io ()
  sigmoid_      :: t -> t -> io ()
  log_          :: t -> t -> io ()
  lgamma_       :: t -> t -> io ()
  log1p_        :: t -> t -> io ()
  exp_          :: t -> t -> io ()
  cos_          :: t -> t -> io ()
  acos_         :: t -> t -> io ()
  cosh_         :: t -> t -> io ()
  sin_          :: t -> t -> io ()
  asin_         :: t -> t -> io ()
  sinh_         :: t -> t -> io ()
  tan_          :: t -> t -> io ()
  atan_         :: t -> t -> io ()
  atan2_        :: t -> t -> t -> io ()
  tanh_         :: t -> t -> io ()
  erf_          :: t -> t -> io ()
  erfinv_       :: t -> t -> io ()
  pow_          :: t -> t -> HsReal t -> io ()
  tpow_         :: t -> HsReal t -> t -> io ()
  sqrt_         :: t -> t -> io ()
  rsqrt_        :: t -> t -> io ()
  ceil_         :: t -> t -> io ()
  floor_        :: t -> t -> io ()
  round_        :: t -> t -> io ()
  trunc_        :: t -> t -> io ()
  frac_         :: t -> t -> io ()
  lerp_         :: t -> t -> t -> HsReal t -> io ()

  mean_         :: t -> t -> Int32 -> Int32 -> io ()
  std_          :: t -> t -> Int32 -> Int32 -> Int32 -> io ()
  var_          :: t -> t -> Int32 -> Int32 -> Int32 -> io ()
  norm_         :: t -> t -> HsReal t -> Int32 -> Int32 -> io ()
  renorm_       :: t -> t -> HsReal t -> Int32 -> HsReal t -> io ()
  dist          :: t -> t -> HsReal t -> io (HsAccReal t)
  meanall       :: t -> io (HsAccReal t)
  varall        :: t -> Int32 -> io (HsAccReal t)
  stdall        :: t -> Int32 -> io (HsAccReal t)
  normall       :: t -> HsReal t -> io (HsAccReal t)

class CPUTensorMathPointwiseFloating t where
  histc_        :: t -> t -> Int64 -> HsReal t -> HsReal t -> io ()
  bhistc_       :: t -> t -> Int64 -> HsReal t -> HsReal t -> io ()


