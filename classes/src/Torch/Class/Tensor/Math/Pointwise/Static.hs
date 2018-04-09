{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Class.Tensor.Math.Pointwise.Static where

import GHC.Int
import System.IO.Unsafe
import Torch.Dimensions

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Class.Tensor.Math.Static (TensorMath)
import qualified Torch.Class.Tensor.Math.Pointwise as Dynamic

class (TensorMath t) => TensorMathPointwise t where
  _sign        :: Dimensions d => t d -> t d -> IO ()
  _clamp       :: Dimensions d => t d -> t d -> HsReal (t d) -> HsReal (t d) -> IO ()
  _cmaxValue   :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()
  _cminValue   :: Dimensions d => t d -> t d -> HsReal (t d) -> IO ()

  _cross       :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> DimVal -> IO ()

  _cadd        :: (Num (HsReal (t d)), Dimensions d) => t d -> t d' -> HsReal (t d) -> t d'' -> IO ()
  _csub        :: (Num (HsReal (t d)), Dimensions d) => t d -> t d' -> HsReal (t d) -> t d'' -> IO ()
  _cmul        :: Dimensions d => t d -> t d' -> t d'' -> IO ()
  _cpow        :: Dimensions d => t d -> t d' -> t d'' -> IO ()
  _cdiv        :: Dimensions d => t d -> t d' -> t d'' -> IO ()

  _clshift     :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()
  _crshift     :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()
  _cfmod       :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()
  _cremainder  :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()
  _cmax        :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()
  _cmin        :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()
  _cbitand     :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()
  _cbitor      :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()
  _cbitxor     :: Dimensions3 d d' d'' => t d -> t d' -> t d'' -> IO ()

  _addcmul     :: Dimensions2 d d' => t d' -> t d -> HsReal (t d) -> t d -> t d -> IO ()
  _addcdiv     :: Dimensions2 d d' => t d' -> t d -> HsReal (t d) -> t d -> t d -> IO ()

sign_, sign :: (TensorMathPointwise t, Dimensions d) => t d -> IO (t d)
sign_ t = withInplace t _sign
sign  t = withEmpty $ \r -> _sign r t

clamp_, clamp :: (TensorMathPointwise t, Dimensions d) => t d -> HsReal (t d) -> HsReal (t d) -> IO (t d)
clamp_ t a b = withInplace t $ \r' t' -> _clamp r' t' a b
clamp  t a b = withEmpty $ \r -> _clamp r t a b

cadd_, cadd :: (Num (HsReal (t d)), TensorMathPointwise t, Dimensions d) => t d -> HsReal (t d) -> t d -> IO (t d)
cadd_ t v b = withInplace t $ \r' t' -> _cadd r' t' v b
cadd  t v b = withEmpty $ \r -> _cadd r t v b
(^+^) :: (Num (HsReal (t d)), TensorMathPointwise t, Dimensions d) => t d -> t d -> t d
(^+^) a b = unsafePerformIO $ cadd a 1 b
{-# NOINLINE (^+^) #-}

csub_, csub :: (Num (HsReal (t d)), TensorMathPointwise t, Dimensions d) => t d -> HsReal (t d) -> t d -> IO (t d)
csub_ t v b = withInplace t $ \r' t' -> _csub r' t' v b
csub  t v b = withEmpty $ \r -> _csub r t v b
(^-^) :: (Num (HsReal (t d)), TensorMathPointwise t, Dimensions d) => t d -> t d -> t d
(^-^) a b = unsafePerformIO $ csub a 1 b
{-# NOINLINE (^-^) #-}

cmul_, cmul :: (TensorMathPointwise t, Dimensions d) => t d -> t d -> IO (t d)
cmul_ t1 t2 = withInplace t1 $ \r' t1' -> _cmul r' t1' t2
cmul  t1 t2 = withEmpty $ \r -> _cmul r t1 t2
(^*^) :: (TensorMathPointwise t, Dimensions d) => t d -> t d -> t d
(^*^) a b = unsafePerformIO $ cmul a b
{-# NOINLINE (^*^) #-}

cdiv_, cdiv :: TensorMathPointwise t => Dimensions d => t d -> t d -> IO (t d)
cdiv_ t1 t2 = withInplace t1 $ \r' t1' -> _cdiv r' t1' t2
cdiv  t1 t2 = withEmpty $ \r -> _cdiv r t1 t2
(^/^) :: (TensorMathPointwise t, Dimensions d) => t d -> t d -> t d
(^/^) a b = unsafePerformIO $ cdiv a b
{-# NOINLINE (^/^) #-}

cpow_, cpow  :: (Dimensions d, TensorMathPointwise t) => t d -> t d -> IO (t d)
cpow_ t1 t2 = withInplace t1 $ \r' t1' -> _cpow r' t1' t2
cpow  t1 t2 = withEmpty $ \r -> _cpow r t1 t2


class IsTensor t => TensorMathPointwiseSigned t where
  _neg :: Dimensions d => t d -> t d -> IO ()
  _abs :: Dimensions d => t d -> t d -> IO ()

neg, abs :: forall t d . (TensorMathPointwiseSigned t, Dimensions d) => t d -> IO (t d)
neg t = withEmpty (`_neg` t)
abs t = withEmpty (`_abs` t)

class TensorMathPointwise t => TensorMathPointwiseFloating t where
  _cinv         :: (Dimensions d) => t d -> t d -> IO ()
  _sigmoid      :: (Dimensions d) => t d -> t d -> IO ()
  _log          :: (Dimensions d) => t d -> t d -> IO ()
  _lgamma       :: (Dimensions d) => t d -> t d -> IO ()
  _log1p        :: (Dimensions d) => t d -> t d -> IO ()
  _exp          :: (Dimensions d) => t d -> t d -> IO ()
  _cos          :: (Dimensions d) => t d -> t d -> IO ()
  _acos         :: (Dimensions d) => t d -> t d -> IO ()
  _cosh         :: (Dimensions d) => t d -> t d -> IO ()
  _sin          :: (Dimensions d) => t d -> t d -> IO ()
  _asin         :: (Dimensions d) => t d -> t d -> IO ()
  _sinh         :: (Dimensions d) => t d -> t d -> IO ()
  _tan          :: (Dimensions d) => t d -> t d -> IO ()
  _atan         :: (Dimensions d) => t d -> t d -> IO ()
  _atan2        :: (Dimensions3 d d' d'') => t d -> t d' -> t d'' -> IO ()
  _tanh         :: (Dimensions d) => t d -> t d -> IO ()
  _erf          :: (Dimensions d) => t d -> t d -> IO ()
  _erfinv       :: (Dimensions d) => t d -> t d -> IO ()
  _pow          :: (Dimensions d) => t d -> t d -> HsReal (t d) -> IO ()
  _tpow         :: (Dimensions d) => t d -> HsReal (t d) -> t d -> IO ()
  _sqrt         :: (Dimensions d) => t d -> t d -> IO ()
  _rsqrt        :: (Dimensions d) => t d -> t d -> IO ()
  _ceil         :: (Dimensions d) => t d -> t d -> IO ()
  _floor        :: (Dimensions d) => t d -> t d -> IO ()
  _round        :: (Dimensions d) => t d -> t d -> IO ()
  _trunc        :: (Dimensions d) => t d -> t d -> IO ()
  _frac         :: (Dimensions d) => t d -> t d -> IO ()
  _lerp         :: (Dimensions3 d d' d'') => t d -> t d' -> t d'' -> HsReal (t d') -> IO ()

cinv_, cinv :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
cinv_ t = withInplace t _cinv
cinv  t = withEmpty $ \r -> _cinv r t

sigmoid_, sigmoid :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
sigmoid_ t = withInplace t _sigmoid
sigmoid  t = withEmpty $ \r -> _sigmoid r t

log_, log :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
log_ t = withInplace t _log
log  t = withEmpty $ \r -> _log r t

lgamma_, lgamma :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
lgamma_ t = withInplace t _lgamma
lgamma  t = withEmpty $ \r -> _lgamma r t

log1p_, log1p :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
log1p_ t = withInplace t _log1p
log1p  t = withEmpty $ \r -> _log1p r t

exp_, exp :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
exp_ t = withInplace t _exp
exp  t = withEmpty $ \r -> _exp r t

cos_, cos :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
cos_ t = withInplace t _cos
cos  t = withEmpty $ \r -> _cos r t

acos_, acos :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
acos_ t = withInplace t _acos
acos  t = withEmpty $ \r -> _acos r t

cosh_, cosh :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
cosh_ t = withInplace t _cosh
cosh  t = withEmpty $ \r -> _cosh r t

sin_, sin :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
sin_ t = withInplace t _sin
sin  t = withEmpty $ \r -> _sin r t

asin_, asin :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
asin_ t = withInplace t _asin
asin  t = withEmpty $ \r -> _asin r t

sinh_, sinh :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
sinh_ t = withInplace t _sinh
sinh  t = withEmpty $ \r -> _sinh r t

tan_, tan :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
tan_ t = withInplace t _tan
tan  t = withEmpty $ \r -> _tan r t

atan_, atan :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
atan_ t = withInplace t _atan
atan  t = withEmpty $ \r -> _atan r t

tanh_, tanh :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
tanh_ t = withInplace t _tanh
tanh  t = withEmpty $ \r -> _tanh r t

erf_, erf :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
erf_ t = withInplace t _erf
erf  t = withEmpty $ \r -> _erf r t

erfinv_, erfinv :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
erfinv_ t = withInplace t _erfinv
erfinv  t = withEmpty $ \r -> _erfinv r t

pow_, pow :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> HsReal (t d) -> IO (t d)
pow_ t v = withInplace t  $ \r t' -> _pow r t' v
pow  t v = withEmpty $ \r -> _pow r t v

tpow_, tpow :: (Dimensions d, TensorMathPointwiseFloating t) => HsReal (t d) -> t d -> IO (t d)
tpow_ v t = withInplace t $ \r t' -> _tpow r v t'
tpow  v t = withEmpty $ \r -> _tpow r v t

sqrt_, sqrt :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
sqrt_ t = withInplace t _sqrt
sqrt  t = withEmpty $ \r -> _sqrt r t

rsqrt_, rsqrt :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
rsqrt_ t = withInplace t _rsqrt
rsqrt  t = withEmpty $ \r -> _rsqrt r t

ceil_, ceil :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
ceil_ t = withInplace t _ceil
ceil  t = withEmpty $ \r -> _ceil r t

floor_, floor :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
floor_ t = withInplace t _floor
floor  t = withEmpty $ \r -> _floor r t

round_, round :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
round_ t = withInplace t _round
round  t = withEmpty $ \r -> _round r t

trunc_, trunc :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
trunc_ t = withInplace t _trunc
trunc  t = withEmpty $ \r -> _trunc r t

frac_, frac :: (Dimensions d, TensorMathPointwiseFloating t) => t d -> IO (t d)
frac_ t = withInplace t _frac
frac  t = withEmpty $ \r -> _frac r t


