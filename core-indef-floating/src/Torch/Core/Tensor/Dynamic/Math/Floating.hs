{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Math.Floating where

import Foreign
import Foreign.C.Types
import GHC.Int
import qualified TensorMathFloating as Sig
import qualified Torch.Class.Tensor.Math as Class
import THTypes

import Torch.Core.Types

instance Class.TensorMathFloating Tensor where
  cinv    = with2Tensors Sig.c_cinv
  sigmoid = with2Tensors Sig.c_sigmoid
  log     = with2Tensors Sig.c_log
  lgamma  = with2Tensors Sig.c_lgamma
  log1p   = with2Tensors Sig.c_log1p
  exp     = with2Tensors Sig.c_exp
  cos     = with2Tensors Sig.c_cos
  acos    = with2Tensors Sig.c_acos
  cosh    = with2Tensors Sig.c_cosh
  sin     = with2Tensors Sig.c_sin
  asin    = with2Tensors Sig.c_asin
  sinh    = with2Tensors Sig.c_sinh
  tan     = with2Tensors Sig.c_tan
  atan    = with2Tensors Sig.c_atan
  atan2   = with3Tensors Sig.c_atan2
  tanh    = with2Tensors Sig.c_tanh
  erf     = with2Tensors Sig.c_erf
  erfinv  = with2Tensors Sig.c_erfinv

  pow :: Tensor -> Tensor -> HsReal -> IO ()
  pow t0 t1 v0 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_pow t0' t1' (hs2cReal v0)

  tpow :: Tensor -> HsReal -> Tensor -> IO ()
  tpow t0 v t1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_tpow t0' (hs2cReal v) t1'

  sqrt   = with2Tensors Sig.c_sqrt
  rsqrt  = with2Tensors Sig.c_rsqrt
  ceil   = with2Tensors Sig.c_ceil
  floor  = with2Tensors Sig.c_floor
  round  = with2Tensors Sig.c_round
  trunc  = with2Tensors Sig.c_trunc
  frac   = with2Tensors Sig.c_frac

  lerp :: Tensor -> Tensor -> Tensor -> HsReal -> IO ()
  lerp t0 t1 t2 v = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> Sig.c_lerp t0' t1' t2' (hs2cReal v)

  mean :: Tensor -> Tensor -> Int32 -> Int32 -> IO ()
  mean t0 t1 i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_mean t0' t1' (CInt i0) (CInt i1)

  std :: Tensor -> Tensor -> Int32 -> Int32 -> Int32 -> IO ()
  std t0 t1 i0 i1 i2 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_std t0' t1' (CInt i0) (CInt i1) (CInt i2)

  var :: Tensor -> Tensor -> Int32 -> Int32 -> Int32 -> IO ()
  var t0 t1 i0 i1 i2 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_var t0' t1' (CInt i0) (CInt i1) (CInt i2)

  norm :: Tensor -> Tensor -> HsReal -> Int32 -> Int32 -> IO ()
  norm t0 t1 v i0 i1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_norm t0' t1' (hs2cReal v) (CInt i0) (CInt i1)

  renorm :: Tensor -> Tensor -> HsReal -> Int32 -> HsReal -> IO ()
  renorm t0 t1 v0 i0 v1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_renorm t0' t1' (hs2cReal v0) (CInt i0) (hs2cReal v1)

  dist :: Tensor -> Tensor -> HsReal -> IO HsAccReal
  dist t0 t1 v0 = _with2Tensors t0 t1 $ \t0' t1' -> pure . c2hsAccReal $ Sig.c_dist t0' t1' (hs2cReal v0)

  histc :: Tensor -> Tensor -> Int64 -> HsReal -> HsReal -> IO ()
  histc t0 t1 l0 v0 v1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_histc t0' t1' (CLLong l0) (hs2cReal v0) (hs2cReal v1)

  bhistc :: Tensor -> Tensor -> Int64 -> HsReal -> HsReal -> IO ()
  bhistc t0 t1 l0 v0 v1 = _with2Tensors t0 t1 $ \t0' t1' -> Sig.c_bhistc t0' t1' (CLLong l0) (hs2cReal v0) (hs2cReal v1)

  meanall :: Tensor -> IO HsAccReal
  meanall = withTensor (pure . c2hsAccReal . Sig.c_meanall)

  varall :: Tensor -> Int32 -> IO HsAccReal
  varall t i = _withTensor t $ pure . c2hsAccReal . (`Sig.c_varall` (CInt i))

  stdall :: Tensor -> Int32 -> IO HsAccReal
  stdall t i = _withTensor t $ pure . c2hsAccReal . (`Sig.c_stdall` (CInt i))

  normall :: Tensor -> HsReal -> IO HsAccReal
  normall t r = _withTensor t $ pure . c2hsAccReal . (`Sig.c_normall` (hs2cReal r))

  linspace :: Tensor -> HsReal -> HsReal -> Int64 -> IO ()
  linspace t r0 r1 l = _withTensor t $ \t' -> Sig.c_linspace t' (hs2cReal r0) (hs2cReal r1) (CLLong l)

  logspace :: Tensor -> HsReal -> HsReal -> Int64 -> IO ()
  logspace t r0 r1 l = _withTensor t $ \t' -> Sig.c_logspace t' (hs2cReal r0) (hs2cReal r1) (CLLong l)

  rand :: Tensor -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()
  rand t g ls = _withTensor t $ \t' -> Sig.c_rand t' g ls

  randn :: Tensor -> Ptr CTHGenerator -> Ptr CTHLongStorage -> IO ()
  randn t g ls = _withTensor t $ \t' -> Sig.c_randn t' g ls

