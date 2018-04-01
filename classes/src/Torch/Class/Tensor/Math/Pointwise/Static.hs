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

class TensorMathPointwiseFloating t where
  cinv_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  sigmoid_      :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  log_          :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  lgamma_       :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  log1p_        :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  exp_          :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  cos_          :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  acos_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  cosh_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  sin_          :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  asin_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  sinh_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  tan_          :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  atan_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  atan2_        :: (Dimensions d, Dimensions d', Dimensions d'') => t d -> t d' -> t d'' -> IO ()
  tanh_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  erf_          :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  erfinv_       :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  pow_          :: (Dimensions d, Dimensions d') => t d -> t d' -> HsReal (t d) -> IO ()
  tpow_         :: (Dimensions d, Dimensions d') => t d -> HsReal (t d') -> t d' -> IO ()
  sqrt_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  rsqrt_        :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  ceil_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  floor_        :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  round_        :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  trunc_        :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  frac_         :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  lerp_         :: (Dimensions d, Dimensions d', Dimensions d'') => t d -> t d' -> t d'' -> HsReal (t d') -> IO ()

{-
class CPUTensorMathPointwiseFloating t d where
  histc_        :: t d -> t d -> Int64 -> HsReal (t d) -> HsReal (t d) -> IO ()
  bhistc_       :: t d -> t d -> Int64 -> HsReal (t d) -> HsReal (t d) -> IO ()


-}
