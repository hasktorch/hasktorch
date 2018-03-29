module Torch.Indef.Dynamic.Tensor.Math.Pairwise where

import Torch.Class.Tensor.Math.Pairwise
import qualified Torch.Sig.Tensor.Math.Pairwise as Sig

import Torch.Indef.Types

instance TensorMathPairwise Dynamic where
  add_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  add_ r t v = with2DynamicState r t $ shuffle3 Sig.c_add (hs2cReal v)

  sub_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  sub_ r t v = with2DynamicState r t $ shuffle3 Sig.c_sub (hs2cReal v)

  add_scaled_ :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  add_scaled_ r t v0 v1 = with2DynamicState r t $ shuffle3'2 Sig.c_add_scaled (hs2cReal v0) (hs2cReal v1)

  sub_scaled_ :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
  sub_scaled_ r t v0 v1 = with2DynamicState r t $ shuffle3'2 Sig.c_sub_scaled (hs2cReal v0) (hs2cReal v1)

  mul_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  mul_ r t v = with2DynamicState r t $ shuffle3 Sig.c_mul (hs2cReal v)

  div_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  div_ r t v = with2DynamicState r t $ shuffle3 Sig.c_div (hs2cReal v)

  lshift_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  lshift_ r t v = with2DynamicState r t $ shuffle3 Sig.c_lshift (hs2cReal v)

  rshift_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  rshift_ r t v = with2DynamicState r t $ shuffle3 Sig.c_rshift (hs2cReal v)

  fmod_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  fmod_ r t v = with2DynamicState r t $ shuffle3 Sig.c_fmod (hs2cReal v)

  remainder_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  remainder_ r t v = with2DynamicState r t $ shuffle3 Sig.c_remainder (hs2cReal v)

  bitand_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  bitand_ r t v = with2DynamicState r t $ shuffle3 Sig.c_bitand (hs2cReal v)

  bitor_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  bitor_ r t v = with2DynamicState r t $ shuffle3 Sig.c_bitor (hs2cReal v)

  bitxor_ :: Dynamic -> Dynamic -> HsReal -> IO ()
  bitxor_ r t v = with2DynamicState r t $ shuffle3 Sig.c_bitxor (hs2cReal v)

  equal :: Dynamic -> Dynamic -> IO Bool
  equal r t = with2DynamicState r t (fmap (== 1) ..: Sig.c_equal)

