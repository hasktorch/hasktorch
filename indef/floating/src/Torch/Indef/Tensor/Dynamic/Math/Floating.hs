{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Tensor.Dynamic.Math.Floating where

import Foreign
import Foreign.C.Types
import GHC.Int
import qualified Torch.Sig.Tensor.MathFloating as Sig
import qualified Torch.Class.Tensor.Math as Class
import Torch.Types.TH
import Torch.Types.TH.Random
import qualified Torch.Types.TH.Long as Long

import Torch.Indef.Types

instance Class.TensorMathFloating Tensor where
  cinv_    = with2Tensors Sig.c_cinv
  sigmoid_ = with2Tensors Sig.c_sigmoid
  log_     = with2Tensors Sig.c_log
  lgamma_  = with2Tensors Sig.c_lgamma
  log1p_   = with2Tensors Sig.c_log1p
  exp_     = with2Tensors Sig.c_exp
  cos_     = with2Tensors Sig.c_cos
  acos_    = with2Tensors Sig.c_acos
  cosh_    = with2Tensors Sig.c_cosh
  sin_     = with2Tensors Sig.c_sin
  asin_    = with2Tensors Sig.c_asin
  sinh_    = with2Tensors Sig.c_sinh
  tan_     = with2Tensors Sig.c_tan
  atan_    = with2Tensors Sig.c_atan
  atan2_   = with3Tensors Sig.c_atan2
  tanh_    = with2Tensors Sig.c_tanh
  erf_     = with2Tensors Sig.c_erf
  erfinv_  = with2Tensors Sig.c_erfinv

  pow_ :: Tensor -> Tensor -> HsReal -> IO ()
  pow_ t0 t1 v0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_pow t0' t1' (hs2cReal v0)

  tpow_ :: Tensor -> HsReal -> Tensor -> IO ()
  tpow_ t0 v t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_tpow t0' (hs2cReal v) t1'

  sqrt_   = with2Tensors Sig.c_sqrt
  rsqrt_  = with2Tensors Sig.c_rsqrt
  ceil_   = with2Tensors Sig.c_ceil
  floor_  = with2Tensors Sig.c_floor
  round_  = with2Tensors Sig.c_round
  trunc_  = with2Tensors Sig.c_trunc
  frac_   = with2Tensors Sig.c_frac

  lerp_ :: Tensor -> Tensor -> Tensor -> HsReal -> IO ()
  lerp_ t0 t1 t2 v = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_lerp t0' t1' t2' (hs2cReal v)

  mean_ :: Tensor -> Tensor -> Int32 -> Int32 -> IO ()
  mean_ t0 t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_mean t0' t1' (CInt i0) (CInt i1)

  std_ :: Tensor -> Tensor -> Int32 -> Int32 -> Int32 -> IO ()
  std_ t0 t1 i0 i1 i2 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_std t0' t1' (CInt i0) (CInt i1) (CInt i2)

  var_ :: Tensor -> Tensor -> Int32 -> Int32 -> Int32 -> IO ()
  var_ t0 t1 i0 i1 i2 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_var t0' t1' (CInt i0) (CInt i1) (CInt i2)

  norm_ :: Tensor -> Tensor -> HsReal -> Int32 -> Int32 -> IO ()
  norm_ t0 t1 v i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_norm t0' t1' (hs2cReal v) (CInt i0) (CInt i1)

  renorm_ :: Tensor -> Tensor -> HsReal -> Int32 -> HsReal -> IO ()
  renorm_ t0 t1 v0 i0 v1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_renorm t0' t1' (hs2cReal v0) (CInt i0) (hs2cReal v1)

  dist :: Tensor -> Tensor -> HsReal -> IO HsAccReal
  dist t0 t1 v0 = _with2Tensors t0 t1 $ \t0' t1' -> fmap c2hsAccReal $ Sig.c_dist t0' t1' (hs2cReal v0)

  histc_ :: Tensor -> Tensor -> Int64 -> HsReal -> HsReal -> IO ()
  histc_ t0 t1 l0 v0 v1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_histc t0' t1' (CLLong l0) (hs2cReal v0) (hs2cReal v1)

  bhistc_ :: Tensor -> Tensor -> Int64 -> HsReal -> HsReal -> IO ()
  bhistc_ t0 t1 l0 v0 v1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_bhistc t0' t1' (CLLong l0) (hs2cReal v0) (hs2cReal v1)

  meanall :: Tensor -> IO HsAccReal
  meanall = withTensor (fmap c2hsAccReal . Sig.c_meanall)

  varall :: Tensor -> Int32 -> IO HsAccReal
  varall t i = _withTensor t $ fmap c2hsAccReal . (`Sig.c_varall` (CInt i))

  stdall :: Tensor -> Int32 -> IO HsAccReal
  stdall t i = _withTensor t $ fmap c2hsAccReal . (`Sig.c_stdall` (CInt i))

  normall :: Tensor -> HsReal -> IO HsAccReal
  normall t r = _withTensor t $ fmap c2hsAccReal . (`Sig.c_normall` (hs2cReal r))

  linspace_ :: Tensor -> HsReal -> HsReal -> Int64 -> IO ()
  linspace_ t r0 r1 l = _withTensor t $ \t' -> Sig.c_linspace t' (hs2cReal r0) (hs2cReal r1) (CLLong l)

  logspace_ :: Tensor -> HsReal -> HsReal -> Int64 -> IO ()
  logspace_ t r0 r1 l = _withTensor t $ \t' -> Sig.c_logspace t' (hs2cReal r0) (hs2cReal r1) (CLLong l)

  rand_ :: Tensor -> Generator -> Long.Storage -> IO ()
  rand_ t g ls = _withTensor t $ \t' -> withForeignPtr (rng g) $ \g' -> withForeignPtr (Long.storage ls) $ \l' -> Sig.c_rand t' g' l'

  randn_ :: Tensor -> Generator -> Long.Storage -> IO ()
  randn_ t g ls = _withTensor t $ \t' -> withForeignPtr (rng g) $ \g' -> withForeignPtr (Long.storage ls) $ \l' -> Sig.c_randn t' g' l'

