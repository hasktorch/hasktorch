module Torch.Indef.Dynamic.Tensor.Math.Pairwise where

import Torch.Indef.Dynamic.Tensor ()
import Torch.Class.Tensor.Math.Pairwise
import qualified Torch.Sig.Tensor.Math.Pairwise as Sig

import Torch.Indef.Types

instance TensorMathPairwise Dynamic where
  _add :: Dynamic -> Dynamic -> HsReal -> IO ()
  _add r t v = with2DynamicState r t $ shuffle3 Sig.c_add (hs2cReal v)

  _sub :: Dynamic -> Dynamic -> HsReal -> IO ()
  _sub r t v = with2DynamicState r t $ shuffle3 Sig.c_sub (hs2cReal v)

  _add_scaled :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  _add_scaled r t v0 v1 = with2DynamicState r t $ shuffle3'2 Sig.c_add_scaled (hs2cReal v0) (hs2cReal v1)

  _sub_scaled :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  _sub_scaled r t v0 v1 = with2DynamicState r t $ shuffle3'2 Sig.c_sub_scaled (hs2cReal v0) (hs2cReal v1)

  _mul :: Dynamic -> Dynamic -> HsReal -> IO ()
  _mul r t v = with2DynamicState r t $ shuffle3 Sig.c_mul (hs2cReal v)

  _div :: Dynamic -> Dynamic -> HsReal -> IO ()
  _div r t v = with2DynamicState r t $ shuffle3 Sig.c_div (hs2cReal v)

  _lshift :: Dynamic -> Dynamic -> HsReal -> IO ()
  _lshift r t v = with2DynamicState r t $ shuffle3 Sig.c_lshift (hs2cReal v)

  _rshift :: Dynamic -> Dynamic -> HsReal -> IO ()
  _rshift r t v = with2DynamicState r t $ shuffle3 Sig.c_rshift (hs2cReal v)

  _fmod :: Dynamic -> Dynamic -> HsReal -> IO ()
  _fmod r t v = with2DynamicState r t $ shuffle3 Sig.c_fmod (hs2cReal v)

  _remainder :: Dynamic -> Dynamic -> HsReal -> IO ()
  _remainder r t v = with2DynamicState r t $ shuffle3 Sig.c_remainder (hs2cReal v)

  _bitand :: Dynamic -> Dynamic -> HsReal -> IO ()
  _bitand r t v = with2DynamicState r t $ shuffle3 Sig.c_bitand (hs2cReal v)

  _bitor :: Dynamic -> Dynamic -> HsReal -> IO ()
  _bitor r t v = with2DynamicState r t $ shuffle3 Sig.c_bitor (hs2cReal v)

  _bitxor :: Dynamic -> Dynamic -> HsReal -> IO ()
  _bitxor r t v = with2DynamicState r t $ shuffle3 Sig.c_bitxor (hs2cReal v)

  equal :: Dynamic -> Dynamic -> IO Bool
  equal r t = with2DynamicState r t (fmap (== 1) ..: Sig.c_equal)

