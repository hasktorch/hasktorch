module Torch.Indef.Static.Tensor.Math.Pointwise.Floating where

import GHC.Int
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating as Dynamic

import Torch.Indef.Types

_cinv a b = Dynamic._cinv (asDynamic a) (asDynamic b)
_sigmoid a b = Dynamic._sigmoid (asDynamic a) (asDynamic b)
_log a b = Dynamic._log (asDynamic a) (asDynamic b)
_lgamma a b = Dynamic._lgamma (asDynamic a) (asDynamic b)
_log1p a b = Dynamic._log1p (asDynamic a) (asDynamic b)
_exp a b = Dynamic._exp (asDynamic a) (asDynamic b)
_cos a b = Dynamic._cos (asDynamic a) (asDynamic b)
_acos a b = Dynamic._acos (asDynamic a) (asDynamic b)
_cosh a b = Dynamic._cosh (asDynamic a) (asDynamic b)
_sin a b = Dynamic._sin (asDynamic a) (asDynamic b)
_asin a b = Dynamic._asin (asDynamic a) (asDynamic b)
_sinh a b = Dynamic._sinh (asDynamic a) (asDynamic b)
_tan a b = Dynamic._tan (asDynamic a) (asDynamic b)
_atan a b = Dynamic._atan (asDynamic a) (asDynamic b)
_tanh a b = Dynamic._tanh (asDynamic a) (asDynamic b)
_erf a b = Dynamic._erf (asDynamic a) (asDynamic b)
_erfinv a b = Dynamic._erfinv (asDynamic a) (asDynamic b)
_sqrt a b = Dynamic._sqrt (asDynamic a) (asDynamic b)
_rsqrt a b = Dynamic._rsqrt (asDynamic a) (asDynamic b)
_ceil a b = Dynamic._ceil (asDynamic a) (asDynamic b)
_floor a b = Dynamic._floor (asDynamic a) (asDynamic b)
_round a b = Dynamic._round (asDynamic a) (asDynamic b)
_trunc a b = Dynamic._trunc (asDynamic a) (asDynamic b)
_frac a b = Dynamic._frac (asDynamic a) (asDynamic b)
_pow a b = Dynamic._pow (asDynamic a) (asDynamic b)
_tpow a v b = Dynamic._tpow (asDynamic a) v (asDynamic b)
_atan2 a b c = Dynamic._atan2 (asDynamic a) (asDynamic b) (asDynamic c)
_lerp a b c = Dynamic._lerp (asDynamic a) (asDynamic b) (asDynamic c)

cinv_, cinv :: (Dimensions d) => Tensor d -> IO (Tensor d)
cinv_ t = withInplace t _cinv
cinv  t = withEmpty $ \r -> _cinv r t

sigmoid_, sigmoid :: (Dimensions d) => Tensor d -> IO (Tensor d)
sigmoid_ t = withInplace t _sigmoid
sigmoid  t = withEmpty $ \r -> _sigmoid r t

log_, log :: (Dimensions d) => Tensor d -> IO (Tensor d)
log_ t = withInplace t _log
log  t = withEmpty $ \r -> _log r t

lgamma_, lgamma :: (Dimensions d) => Tensor d -> IO (Tensor d)
lgamma_ t = withInplace t _lgamma
lgamma  t = withEmpty $ \r -> _lgamma r t

log1p_, log1p :: (Dimensions d) => Tensor d -> IO (Tensor d)
log1p_ t = withInplace t _log1p
log1p  t = withEmpty $ \r -> _log1p r t

exp_, exp :: (Dimensions d) => Tensor d -> IO (Tensor d)
exp_ t = withInplace t _exp
exp  t = withEmpty $ \r -> _exp r t

cos_, cos :: (Dimensions d) => Tensor d -> IO (Tensor d)
cos_ t = withInplace t _cos
cos  t = withEmpty $ \r -> _cos r t

acos_, acos :: (Dimensions d) => Tensor d -> IO (Tensor d)
acos_ t = withInplace t _acos
acos  t = withEmpty $ \r -> _acos r t

cosh_, cosh :: (Dimensions d) => Tensor d -> IO (Tensor d)
cosh_ t = withInplace t _cosh
cosh  t = withEmpty $ \r -> _cosh r t

sin_, sin :: (Dimensions d) => Tensor d -> IO (Tensor d)
sin_ t = withInplace t _sin
sin  t = withEmpty $ \r -> _sin r t

asin_, asin :: (Dimensions d) => Tensor d -> IO (Tensor d)
asin_ t = withInplace t _asin
asin  t = withEmpty $ \r -> _asin r t

sinh_, sinh :: (Dimensions d) => Tensor d -> IO (Tensor d)
sinh_ t = withInplace t _sinh
sinh  t = withEmpty $ \r -> _sinh r t

tan_, tan :: (Dimensions d) => Tensor d -> IO (Tensor d)
tan_ t = withInplace t _tan
tan  t = withEmpty $ \r -> _tan r t

atan_, atan :: (Dimensions d) => Tensor d -> IO (Tensor d)
atan_ t = withInplace t _atan
atan  t = withEmpty $ \r -> _atan r t

tanh_, tanh :: (Dimensions d) => Tensor d -> IO (Tensor d)
tanh_ t = withInplace t _tanh
tanh  t = withEmpty $ \r -> _tanh r t

erf_, erf :: (Dimensions d) => Tensor d -> IO (Tensor d)
erf_ t = withInplace t _erf
erf  t = withEmpty $ \r -> _erf r t

erfinv_, erfinv :: (Dimensions d) => Tensor d -> IO (Tensor d)
erfinv_ t = withInplace t _erfinv
erfinv  t = withEmpty $ \r -> _erfinv r t

pow_, pow :: (Dimensions d) => Tensor d -> HsReal -> IO (Tensor d)
pow_ t v = withInplace t  $ \r t' -> _pow r t' v
pow  t v = withEmpty $ \r -> _pow r t v

tpow_, tpow :: (Dimensions d) => HsReal -> Tensor d -> IO (Tensor d)
tpow_ v t = withInplace t $ \r t' -> _tpow r v t'
tpow  v t = withEmpty $ \r -> _tpow r v t

sqrt_, sqrt :: (Dimensions d) => Tensor d -> IO (Tensor d)
sqrt_ t = withInplace t _sqrt
sqrt  t = withEmpty $ \r -> _sqrt r t

rsqrt_, rsqrt :: (Dimensions d) => Tensor d -> IO (Tensor d)
rsqrt_ t = withInplace t _rsqrt
rsqrt  t = withEmpty $ \r -> _rsqrt r t

ceil_, ceil :: (Dimensions d) => Tensor d -> IO (Tensor d)
ceil_ t = withInplace t _ceil
ceil  t = withEmpty $ \r -> _ceil r t

floor_, floor :: (Dimensions d) => Tensor d -> IO (Tensor d)
floor_ t = withInplace t _floor
floor  t = withEmpty $ \r -> _floor r t

round_, round :: (Dimensions d) => Tensor d -> IO (Tensor d)
round_ t = withInplace t _round
round  t = withEmpty $ \r -> _round r t

trunc_, trunc :: (Dimensions d) => Tensor d -> IO (Tensor d)
trunc_ t = withInplace t _trunc
trunc  t = withEmpty $ \r -> _trunc r t

frac_, frac :: (Dimensions d) => Tensor d -> IO (Tensor d)
frac_ t = withInplace t _frac
frac  t = withEmpty $ \r -> _frac r t

lerp_, lerp :: (Dimensions3 d d' d'') => Tensor d -> Tensor d' -> HsReal -> IO (Tensor d'')
lerp_ a b v = _lerp a a b v >> pure (asStatic (asDynamic a))
lerp  a b v = withEmpty $ \r -> _lerp r a b v

atan2_, atan2 :: (Dimensions3 d d' d'') => Tensor d -> Tensor d' -> IO (Tensor d'')
atan2_ a b = _atan2 a a b >> pure (asStatic (asDynamic a))
atan2  a b = withEmpty $ \r -> _atan2 r a b
