{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Class.Tensor.Math.Pointwise.Static where

import GHC.Int
import Torch.Dimensions

import Torch.Class.Types
import Torch.Class.Tensor.Static
import qualified Torch.Class.Tensor.Math.Pointwise as Dynamic

class TensorMathPointwise t where
  sign_        :: Dimensions d => t d -> t d -> IO ()
  cross_       :: Dimensions d => t d -> t d -> t d -> DimVal -> IO ()
  clamp_       :: Dimensions d => t d -> t d -> HsReal (t d) -> HsReal (t d) -> IO ()
  cadd_        :: Dimensions d => t d -> t d -> HsReal (t d) -> t d -> IO ()
  csub_        :: Dimensions d => t d -> t d -> HsReal (t d) -> t d -> IO ()
  cmul_        :: Dimensions d => t d -> t d -> t d -> IO ()
  cpow_        :: Dimensions d => t d -> t d -> t d -> IO ()
  cdiv_        :: Dimensions d => t d -> t d -> t d -> IO ()
  clshift_     :: Dimensions d => t d -> t d -> t d -> IO ()
  crshift_     :: Dimensions d => t d -> t d -> t d -> IO ()
  cfmod_       :: Dimensions d => t d -> t d -> t d -> IO ()
  cremainder_  :: Dimensions d => t d -> t d -> t d -> IO ()
  cmax_        :: Dimensions d => t d -> t d -> t d -> IO ()
  cmin_        :: Dimensions d => t d -> t d -> t d -> IO ()
  cmaxValue_   :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  cminValue_   :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  cbitand_     :: Dimensions d => t d -> t d -> t d -> IO ()
  cbitor_      :: Dimensions d => t d -> t d -> t d -> IO ()
  cbitxor_     :: Dimensions d => t d -> t d -> t d -> IO ()
  addcmul_     :: Dimensions d => t d -> t d -> HsReal (t d) -> t d -> t d -> IO ()
  addcdiv_     :: Dimensions d => t d -> t d -> HsReal (t d) -> t d -> t d -> IO ()

sign :: forall t d .  (TensorMathPointwise t, Dimensions d, Tensor t) => t d -> IO (t d)
sign t = withInplace (`sign_` t)

class TensorMathPointwiseSigned t where
  neg_ :: Dimensions d => t d -> t d -> IO ()
  abs_ :: Dimensions d => t d -> t d -> IO ()

neg, abs :: forall t d . (TensorMathPointwiseSigned t, Dimensions d, Tensor t) => t d -> IO (t d)
neg t = withInplace (`neg_` t)
abs t = withInplace (`abs_` t)

{-
class TensorMathPointwiseFloating t d where
  cinv_         :: t d -> t d -> IO ()
  sigmoid_      :: t d -> t d -> IO ()
  log_          :: t d -> t d -> IO ()
  lgamma_       :: t d -> t d -> IO ()
  log1p_        :: t d -> t d -> IO ()
  exp_          :: t d -> t d -> IO ()
  cos_          :: t d -> t d -> IO ()
  acos_         :: t d -> t d -> IO ()
  cosh_         :: t d -> t d -> IO ()
  sin_          :: t d -> t d -> IO ()
  asin_         :: t d -> t d -> IO ()
  sinh_         :: t d -> t d -> IO ()
  tan_          :: t d -> t d -> IO ()
  atan_         :: t d -> t d -> IO ()
  atan2_        :: t d -> t d -> t d -> IO ()
  tanh_         :: t d -> t d -> IO ()
  erf_          :: t d -> t d -> IO ()
  erfinv_       :: t d -> t d -> IO ()
  pow_          :: t d -> t d -> HsReal (t d) -> IO ()
  tpow_         :: t d -> HsReal (t d) -> t d -> IO ()
  sqrt_         :: t d -> t d -> IO ()
  rsqrt_        :: t d -> t d -> IO ()
  ceil_         :: t d -> t d -> IO ()
  floor_        :: t d -> t d -> IO ()
  round_        :: t d -> t d -> IO ()
  trunc_        :: t d -> t d -> IO ()
  frac_         :: t d -> t d -> IO ()
  lerp_         :: t d -> t d -> t d -> HsReal (t d) -> IO ()

  mean_         :: t d -> t d -> Int -> Int -> IO ()
  std_          :: t d -> t d -> Int -> Int -> Int -> IO ()
  var_          :: t d -> t d -> Int -> Int -> Int -> IO ()
  norm_         :: t d -> t d -> HsReal (t d) -> Int -> Int -> IO ()
  renorm_       :: t d -> t d -> HsReal (t d) -> Int -> HsReal (t d) -> IO ()
  dist          :: t d -> t d -> HsReal (t d) -> IO (HsAccReal t)
  meanall       :: t d -> IO (HsAccReal t)
  varall        :: t d -> Int -> IO (HsAccReal t)
  stdall        :: t d -> Int -> IO (HsAccReal t)
  normall       :: t d -> HsReal (t d) -> IO (HsAccReal t)

class CPUTensorMathPointwiseFloating t d where
  histc_        :: t d -> t d -> Int64 -> HsReal (t d) -> HsReal (t d) -> IO ()
  bhistc_       :: t d -> t d -> Int64 -> HsReal (t d) -> HsReal (t d) -> IO ()


-}
