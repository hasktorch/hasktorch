module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating where

import GHC.Int
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.Tensor.Math.Pointwise.Floating as Sig

import Torch.Indef.Types

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

cinv_, cinv :: Dynamic -> IO Dynamic
cinv_ t = twice t _cinv
cinv  t = withEmpty t $ \r -> _cinv r t

sigmoid_, sigmoid :: Dynamic -> IO Dynamic
sigmoid_ t = twice t _sigmoid
sigmoid  t = withEmpty t $ \r -> _sigmoid r t

log_, log :: Dynamic -> IO Dynamic
log_ t = twice t _log
log  t = withEmpty t $ \r -> _log r t

lgamma_, lgamma :: Dynamic -> IO Dynamic
lgamma_ t = twice t _lgamma
lgamma  t = withEmpty t $ \r -> _lgamma r t

log1p_, log1p :: Dynamic -> IO Dynamic
log1p_ t = twice t _log1p
log1p  t = withEmpty t $ \r -> _log1p r t

exp_, exp :: Dynamic -> IO Dynamic
exp_ t = twice t _exp
exp  t = withEmpty t $ \r -> _exp r t

cos_, cos :: Dynamic -> IO Dynamic
cos_ t = twice t _cos
cos  t = withEmpty t $ \r -> _cos r t

acos_, acos :: Dynamic -> IO Dynamic
acos_ t = twice t _acos
acos  t = withEmpty t $ \r -> _acos r t

cosh_, cosh :: Dynamic -> IO Dynamic
cosh_ t = twice t _cosh
cosh  t = withEmpty t $ \r -> _cosh r t

sin_, sin :: Dynamic -> IO Dynamic
sin_ t = twice t _sin
sin  t = withEmpty t $ \r -> _sin r t

asin_, asin :: Dynamic -> IO Dynamic
asin_ t = twice t _asin
asin  t = withEmpty t $ \r -> _asin r t

sinh_, sinh :: Dynamic -> IO Dynamic
sinh_ t = twice t _sinh
sinh  t = withEmpty t $ \r -> _sinh r t

tan_, tan :: Dynamic -> IO Dynamic
tan_ t = twice t _tan
tan  t = withEmpty t $ \r -> _tan r t

atan_, atan :: Dynamic -> IO Dynamic
atan_ t = twice t _atan
atan  t = withEmpty t $ \r -> _atan r t

tanh_, tanh :: Dynamic -> IO Dynamic
tanh_ t = twice t _tanh
tanh  t = withEmpty t $ \r -> _tanh r t

erf_, erf :: Dynamic -> IO Dynamic
erf_ t = twice t _erf
erf  t = withEmpty t $ \r -> _erf r t

erfinv_, erfinv :: Dynamic -> IO Dynamic
erfinv_ t = twice t _erfinv
erfinv  t = withEmpty t $ \r -> _erfinv r t

pow_, pow :: Dynamic -> HsReal -> IO Dynamic
pow_ t v = twice t  $ \r t' -> _pow r t' v
pow  t v = withEmpty t $ \r -> _pow r t v

tpow_, tpow :: HsReal -> Dynamic -> IO Dynamic
tpow_ v t = twice t $ \r t' -> _tpow r v t'
tpow  v t = withEmpty t $ \r -> _tpow r v t

sqrt_, sqrt :: Dynamic -> IO Dynamic
sqrt_ t = twice t _sqrt
sqrt  t = withEmpty t $ \r -> _sqrt r t

rsqrt_, rsqrt :: Dynamic -> IO Dynamic
rsqrt_ t = twice t _rsqrt
rsqrt  t = withEmpty t $ \r -> _rsqrt r t

ceil_, ceil :: Dynamic -> IO Dynamic
ceil_ t = twice t _ceil
ceil  t = withEmpty t $ \r -> _ceil r t

floor_, floor :: Dynamic -> IO Dynamic
floor_ t = twice t _floor
floor  t = withEmpty t $ \r -> _floor r t

round_, round :: Dynamic -> IO Dynamic
round_ t = twice t _round
round  t = withEmpty t $ \r -> _round r t

trunc_, trunc :: Dynamic -> IO Dynamic
trunc_ t = twice t _trunc
trunc  t = withEmpty t $ \r -> _trunc r t

frac_, frac :: Dynamic -> IO Dynamic
frac_ t = twice t _frac
frac  t = withEmpty t $ \r -> _frac r t

lerp_, lerp :: Dynamic -> Dynamic -> HsReal -> IO Dynamic
lerp_ a b v = twice a $ \r a' -> _lerp r a' b v
lerp  a b v = withEmpty a $ \r -> _lerp r a b v

atan2_, atan2 :: Dynamic -> Dynamic -> IO Dynamic
atan2_ a b = twice a $ \r a' -> _atan2 r a' b
atan2  a b = withEmpty a $ \r -> _atan2 r a b


-- class CPUTensorMathPointwiseFloating t where
--   histc_        :: Dynamic -> Dynamic -> Int64 -> HsReal -> HsReal -> IO ()
--   bhistc_       :: Dynamic -> Dynamic -> Int64 -> HsReal -> HsReal -> IO ()


