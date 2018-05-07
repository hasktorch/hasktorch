module Torch.Indef.Dynamic.Tensor.Math.Pairwise where

import qualified Torch.Sig.Tensor.Math.Pairwise as Sig

import Torch.Indef.Dynamic.Tensor
import Torch.Indef.Types

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

add_, add :: Dynamic -> HsReal -> IO Dynamic
add_ t v = twice t $ \r t' -> _add r t' v
add  t v = withEmpty t $ \r -> _add r t v

sub_, sub :: Dynamic -> HsReal -> IO Dynamic
sub_ t v = twice t $ \r t' -> _sub r t' v
sub  t v = withEmpty t $ \r -> _sub r t v


add_scaled_ :: Dynamic -> HsReal -> HsReal -> IO Dynamic
add_scaled_ t v0 v1 = twice t $ \r t' -> _add_scaled r t' v0 v1

sub_scaled_ :: Dynamic -> HsReal -> HsReal -> IO Dynamic
sub_scaled_ t v0 v1 = twice t $ \r t' -> _sub_scaled r t' v0 v1

mul_, mul :: Dynamic -> HsReal -> IO Dynamic
mul_ t v = twice t $ \r t' -> _mul r t' v
mul  t v = withEmpty t $ \r -> _mul r t v

div_, div :: Dynamic -> HsReal -> IO Dynamic
div_ t v = twice t $ \r t' -> _div r t' v
div  t v = withEmpty t $ \r -> _div r t v

lshift_ :: Dynamic -> HsReal -> IO Dynamic
lshift_ t v = twice t $ \r t' -> _lshift r t' v

rshift_ :: Dynamic -> HsReal -> IO Dynamic
rshift_ t v = twice t $ \r t' -> _rshift r t' v

fmod_ :: Dynamic -> HsReal -> IO Dynamic
fmod_ t v = twice t $ \r t' -> _fmod r t' v

remainder_ :: Dynamic -> HsReal -> IO Dynamic
remainder_ t v = twice t $ \r t' -> _remainder r t' v

bitand_ :: Dynamic -> HsReal -> IO Dynamic
bitand_ t v = twice t $ \r t' -> _bitand r t' v

bitor_ :: Dynamic -> HsReal -> IO Dynamic
bitor_ t v = twice t $ \r t' -> _bitor r t' v

bitxor_ :: Dynamic -> HsReal -> IO Dynamic
bitxor_ t v = twice t $ \r t' -> _bitxor r t' v


