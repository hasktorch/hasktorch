module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating where

import GHC.Int
import qualified Torch.Class.Tensor.Math.Pointwise as Class
import qualified Torch.Sig.Tensor.Math.Pointwise.Floating as Sig
import Torch.Indef.Dynamic.Tensor ()

import Torch.Indef.Types

instance Class.TensorMathPointwiseFloating Dynamic where
  _cinv :: Dynamic -> Dynamic -> IO ()
  _cinv a b = with2DynamicState a b Sig.c_cinv

  _sigmoid :: Dynamic -> Dynamic -> IO ()
  _sigmoid a b = with2DynamicState a b Sig.c_sigmoid

  _log :: Dynamic -> Dynamic -> IO ()
  _log a b = with2DynamicState a b Sig.c_log

  _lgamma :: Dynamic -> Dynamic -> IO ()
  _lgamma a b = with2DynamicState a b Sig.c_lgamma

  _log1p :: Dynamic -> Dynamic -> IO ()
  _log1p a b = with2DynamicState a b Sig.c_log1p

  _exp :: Dynamic -> Dynamic -> IO ()
  _exp a b = with2DynamicState a b Sig.c_exp

  _cos :: Dynamic -> Dynamic -> IO ()
  _cos a b = with2DynamicState a b Sig.c_cos

  _acos :: Dynamic -> Dynamic -> IO ()
  _acos a b = with2DynamicState a b Sig.c_acos

  _cosh :: Dynamic -> Dynamic -> IO ()
  _cosh a b = with2DynamicState a b Sig.c_cosh

  _sin :: Dynamic -> Dynamic -> IO ()
  _sin a b = with2DynamicState a b Sig.c_sin

  _asin :: Dynamic -> Dynamic -> IO ()
  _asin a b = with2DynamicState a b Sig.c_asin

  _sinh :: Dynamic -> Dynamic -> IO ()
  _sinh a b = with2DynamicState a b Sig.c_sinh

  _tan :: Dynamic -> Dynamic -> IO ()
  _tan a b = with2DynamicState a b Sig.c_tan

  _atan :: Dynamic -> Dynamic -> IO ()
  _atan a b = with2DynamicState a b Sig.c_atan

  _tanh :: Dynamic -> Dynamic -> IO ()
  _tanh a b = with2DynamicState a b Sig.c_tanh

  _erf :: Dynamic -> Dynamic -> IO ()
  _erf a b = with2DynamicState a b Sig.c_erf

  _erfinv :: Dynamic -> Dynamic -> IO ()
  _erfinv a b = with2DynamicState a b Sig.c_erfinv

  _sqrt :: Dynamic -> Dynamic -> IO ()
  _sqrt a b = with2DynamicState a b Sig.c_sqrt

  _rsqrt :: Dynamic -> Dynamic -> IO ()
  _rsqrt a b = with2DynamicState a b Sig.c_rsqrt

  _ceil :: Dynamic -> Dynamic -> IO ()
  _ceil a b = with2DynamicState a b Sig.c_ceil

  _floor :: Dynamic -> Dynamic -> IO ()
  _floor a b = with2DynamicState a b Sig.c_floor

  _round :: Dynamic -> Dynamic -> IO ()
  _round a b = with2DynamicState a b Sig.c_round

  _trunc :: Dynamic -> Dynamic -> IO ()
  _trunc a b = with2DynamicState a b Sig.c_trunc

  _frac :: Dynamic -> Dynamic -> IO ()
  _frac a b = with2DynamicState a b Sig.c_frac

  _atan2 :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _atan2 a b c = with3DynamicState a b c Sig.c_atan2

  _pow :: Dynamic -> Dynamic -> HsReal -> IO ()
  _pow a b v = with2DynamicState a b (shuffle3 Sig.c_pow (hs2cReal v))

  _tpow :: Dynamic -> HsReal -> Dynamic -> IO ()
  _tpow a v b = with2DynamicState a b $ \s' a' b' -> Sig.c_tpow s' a' (hs2cReal v) b'

  _lerp :: Dynamic -> Dynamic -> Dynamic -> HsReal -> IO ()
  _lerp a b c v = with3DynamicState a b c $ \s' a' b' c' -> Sig.c_lerp s' a' b' c' (hs2cReal v)
 
