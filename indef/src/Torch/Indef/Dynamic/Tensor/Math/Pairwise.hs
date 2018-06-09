-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Pairwise
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Math.Pairwise
  ( equal
  , add, add_, add_scaled_
  , sub, sub_, sub_scaled_
  , mul, mul_
  , Torch.Indef.Dynamic.Tensor.Math.Pairwise.div, div_
  , lshift_
  , rshift_
  , fmod_
  , remainder_
  , bitand_
  , bitor_
  , bitxor_
  ) where

import Torch.Indef.Dynamic.Tensor
import Torch.Indef.Types

import qualified Torch.Sig.Tensor.Math.Pairwise as Sig

-- | Call Torch's C-level @equal@ function.
equal :: Dynamic -> Dynamic -> IO Bool
equal r t = with2DynamicState r t (fmap (== 1) ..: Sig.c_equal)

-- | add a scalar to a tensor, inplace.
add_ :: Dynamic -> HsReal -> IO ()
add_ t v = _add t t v

-- | add a scalar to a tensor.
add :: Dynamic -> HsReal -> IO Dynamic
add  t v = withEmpty t $ \r -> _add r t v

-- | subtract a scalar from a tensor, inplace.
sub_ :: Dynamic -> HsReal -> IO ()
sub_ t v = _sub t t v

-- | subtract a scalar from a tensor.
sub :: Dynamic -> HsReal -> IO Dynamic
sub  t v = withEmpty t $ \r -> _sub r t v

-- | add a scalar, which has been scaled, to a tensor, inplace.
add_scaled_
  :: Dynamic  -- ^ tensor to scale
  -> HsReal   -- ^ value to add
  -> HsReal   -- ^ amount to scale the value by
  -> IO ()
add_scaled_ t v0 v1 = _add_scaled t t v0 v1

-- | subtract a scalar, which has been scaled, from a tensor, inplace.
sub_scaled_
  :: Dynamic  -- ^ tensor to scale
  -> HsReal   -- ^ value to add
  -> HsReal   -- ^ amount to scale the value by
  -> IO ()
sub_scaled_ t v0 v1 = _sub_scaled t t v0 v1

-- | multiply a tensor by a scalar value, inplace.
mul_ :: Dynamic -> HsReal -> IO ()
mul_ t v = _mul t t v

-- | multiply a tensor by a scalar value, pure.
mul :: Dynamic -> HsReal -> IO Dynamic
mul t v = withEmpty t $ \r -> _mul r t v

-- | divide a tensor by a scalar value, inplace.
div_ :: Dynamic -> HsReal -> IO ()
div_ t v = _div t t v

-- | divide a tensor by a scalar value, pure.
div :: Dynamic -> HsReal -> IO Dynamic
div t v = withEmpty t $ \r -> _div r t v

-- | Left shift all elements in the tensor by the given value, inplace.
lshift_ :: Dynamic -> HsReal -> IO ()
lshift_ t v = _lshift t t v

-- | Right shift all elements in the tensor by the given value, inplace.
rshift_ :: Dynamic -> HsReal -> IO ()
rshift_ t v = _rshift t t v

-- | Compute the remainder of division ( rounded towards zero) of all elements in the tensor by a given value, inplace.
fmod_ :: Dynamic -> HsReal -> IO ()
fmod_ t v = _fmod t t v

-- | Computes remainder of division (rounded to nearest) of all elements in the tensor by value, inplace
remainder_ :: Dynamic -> HsReal -> IO ()
remainder_ t v = _remainder t t v

-- | Performs the bitwise operation inplace on all elements in the tensor.
bitand_, bitor_, bitxor_ :: Dynamic -> HsReal -> IO ()
bitand_ t v = _bitand t t v
bitor_  t v = _bitor t t v
bitxor_ t v = _bitxor t t v

-- The remainder of this module includes C-styled versions of the haskell API

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


