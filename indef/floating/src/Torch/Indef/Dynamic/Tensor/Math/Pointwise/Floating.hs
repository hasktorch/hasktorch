module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating where

import GHC.Int
import qualified Torch.Class.Tensor.Math.Pointwise as Class
import qualified Torch.Sig.Tensor.Math.Pointwise.Floating as Sig

import Torch.Indef.Types

instance Class.TensorMathPointwiseFloating Dynamic where
  cinv_ :: Dynamic -> Dynamic -> IO ()
  cinv_ a b = with2DynamicState a b Sig.c_cinv

  sigmoid_ :: Dynamic -> Dynamic -> IO ()
  sigmoid_ a b = with2DynamicState a b Sig.c_sigmoid

  log_ :: Dynamic -> Dynamic -> IO ()
  log_ a b = with2DynamicState a b Sig.c_log

  lgamma_ :: Dynamic -> Dynamic -> IO ()
  lgamma_ a b = with2DynamicState a b Sig.c_lgamma

  log1p_ :: Dynamic -> Dynamic -> IO ()
  log1p_ a b = with2DynamicState a b Sig.c_log1p

  exp_ :: Dynamic -> Dynamic -> IO ()
  exp_ a b = with2DynamicState a b Sig.c_exp

  cos_ :: Dynamic -> Dynamic -> IO ()
  cos_ a b = with2DynamicState a b Sig.c_cos

  acos_ :: Dynamic -> Dynamic -> IO ()
  acos_ a b = with2DynamicState a b Sig.c_acos

  cosh_ :: Dynamic -> Dynamic -> IO ()
  cosh_ a b = with2DynamicState a b Sig.c_cosh

  sin_ :: Dynamic -> Dynamic -> IO ()
  sin_ a b = with2DynamicState a b Sig.c_sin

  asin_ :: Dynamic -> Dynamic -> IO ()
  asin_ a b = with2DynamicState a b Sig.c_asin

  sinh_ :: Dynamic -> Dynamic -> IO ()
  sinh_ a b = with2DynamicState a b Sig.c_sinh

  tan_ :: Dynamic -> Dynamic -> IO ()
  tan_ a b = with2DynamicState a b Sig.c_tan

  atan_ :: Dynamic -> Dynamic -> IO ()
  atan_ a b = with2DynamicState a b Sig.c_atan

  tanh_ :: Dynamic -> Dynamic -> IO ()
  tanh_ a b = with2DynamicState a b Sig.c_tanh

  erf_ :: Dynamic -> Dynamic -> IO ()
  erf_ a b = with2DynamicState a b Sig.c_erf

  erfinv_ :: Dynamic -> Dynamic -> IO ()
  erfinv_ a b = with2DynamicState a b Sig.c_erfinv

  sqrt_ :: Dynamic -> Dynamic -> IO ()
  sqrt_ a b = with2DynamicState a b Sig.c_sqrt

  rsqrt_ :: Dynamic -> Dynamic -> IO ()
  rsqrt_ a b = with2DynamicState a b Sig.c_rsqrt

  ceil_ :: Dynamic -> Dynamic -> IO ()
  ceil_ a b = with2DynamicState a b Sig.c_ceil

  floor_ :: Dynamic -> Dynamic -> IO ()
  floor_ a b = with2DynamicState a b Sig.c_floor

  round_ :: Dynamic -> Dynamic -> IO ()
  round_ a b = with2DynamicState a b Sig.c_round

  trunc_ :: Dynamic -> Dynamic -> IO ()
  trunc_ a b = with2DynamicState a b Sig.c_trunc

  frac_ :: Dynamic -> Dynamic -> IO ()
  frac_ a b = with2DynamicState a b Sig.c_frac

  atan2_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  atan2_ a b c = with3DynamicState a b c Sig.c_atan2

  pow_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  pow_ a b v = with2DynamicState a b (shuffle3 Sig.c_pow (hs2cReal v))

  tpow_ :: Dynamic -> HsReal -> Dynamic -> IO ()
  tpow_ a v b = with2DynamicState a b $ \s' a' b' -> Sig.c_tpow s' a' (hs2cReal v) b'

  lerp_ :: Dynamic -> Dynamic -> Dynamic -> HsReal -> IO ()
  lerp_ a b c v = with3DynamicState a b c $ \s' a' b' c' -> Sig.c_lerp s' a' b' c' (hs2cReal v)
 
