{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Class.Tensor.Math.Pointwise where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Torch.Class.Types
import Torch.Dimensions
import Torch.Class.Tensor
import System.IO.Unsafe

class IsTensor t => TensorMathPointwise t where
  _sign        :: t -> t -> IO ()
  _cross       :: t -> t -> t -> DimVal -> IO ()
  _clamp       :: t -> t -> HsReal t -> HsReal t -> IO ()
  _cadd        :: t -> t -> HsReal t -> t -> IO ()
  _csub        :: t -> t -> HsReal t -> t -> IO ()
  _cmul        :: t -> t -> t -> IO ()
  _cpow        :: t -> t -> t -> IO ()
  _cdiv        :: t -> t -> t -> IO ()
  _clshift     :: t -> t -> t -> IO ()
  _crshift     :: t -> t -> t -> IO ()
  _cfmod       :: t -> t -> t -> IO ()
  _cremainder  :: t -> t -> t -> IO ()
  _cmax        :: t -> t -> t -> IO ()
  _cmin        :: t -> t -> t -> IO ()
  _cmaxValue   :: t -> t -> HsReal t -> IO ()
  _cminValue   :: t -> t -> HsReal t -> IO ()
  _cbitand     :: t -> t -> t -> IO ()
  _cbitor      :: t -> t -> t -> IO ()
  _cbitxor     :: t -> t -> t -> IO ()
  _addcmul     :: t -> t -> HsReal t -> t -> t -> IO ()
  _addcdiv     :: t -> t -> HsReal t -> t -> t -> IO ()

sign_, sign :: TensorMathPointwise t => t -> IO t
sign_  = (`twice` _sign)
sign t = withEmpty $ \r -> _sign r t

cross :: TensorMathPointwise t => t -> t -> DimVal -> IO t
cross a b di = withEmpty $ \r -> _cross r a b di

clamp_, clamp :: TensorMathPointwise t => t -> HsReal t -> HsReal t -> IO t
clamp_ t a b = t `twice` (\r' t' -> _clamp r' t' a b)
clamp  t a b = withEmpty $ \r -> _clamp r t a b

cadd_, cadd :: TensorMathPointwise t => t -> HsReal t -> t -> IO t
cadd_ t v b = t `twice` (\r' t' -> _cadd r' t' v b)
cadd  t v b = withEmpty $ \r -> _cadd r t v b
(^+^) :: (Num (HsReal t), TensorMathPointwise t) => t -> t -> t
(^+^) a b = unsafePerformIO $ cadd a 1 b
{-# NOINLINE (^+^) #-}

csub_, csub :: TensorMathPointwise t => t -> HsReal t -> t -> IO t
csub_ t v b = t `twice` (\r' t' -> _csub r' t' v b)
csub  t v b = withEmpty $ \r -> _csub r t v b
(^-^) :: (Num (HsReal t), TensorMathPointwise t) => t -> t -> t
(^-^) a b = unsafePerformIO $ csub a 1 b
{-# NOINLINE (^-^) #-}

cmul_, cmul :: TensorMathPointwise t => t -> t -> IO t
cmul_ t1 t2 = t1 `twice` (\r' t1' -> _cmul r' t1' t2)
cmul  t1 t2 = withEmpty $ \r -> _cmul r t1 t2
square :: TensorMathPointwise t => t -> IO t
square t = cmul t t
(^*^) :: TensorMathPointwise t => t -> t -> t
(^*^) a b = unsafePerformIO $ cmul a b
{-# NOINLINE (^*^) #-}

cdiv_, cdiv :: TensorMathPointwise t => t -> t -> IO t
cdiv_ t1 t2 = t1 `twice` (\r' t1' -> _cdiv r' t1' t2)
cdiv  t1 t2 = withEmpty $ \r -> _cdiv r t1 t2
(^/^) :: TensorMathPointwise t => t -> t -> t
(^/^) a b = unsafePerformIO $ cdiv a b
{-# NOINLINE (^/^) #-}

cpow_, cpow  :: TensorMathPointwise t => t -> t -> IO t
cpow_ t1 t2 = t1 `twice` (\r' t1' -> _cpow r' t1' t2)
cpow  t1 t2 = withEmpty $ \r -> _cpow r t1 t2


class IsTensor t => TensorMathPointwiseSigned t where
  _neg :: t -> t -> IO ()
  _abs :: t -> t -> IO ()

neg_, neg  :: TensorMathPointwiseSigned t => t -> IO t
neg_ t = t `twice` _neg
neg  t = withEmpty $ \r -> _neg r t

abs_, abs  :: TensorMathPointwiseSigned t => t -> IO t
abs_ t = t `twice` _abs
abs  t = withEmpty $ \r -> _abs r t

class IsTensor t => TensorMathPointwiseFloating t where
  _cinv         :: t -> t -> IO ()
  _sigmoid      :: t -> t -> IO ()
  _log          :: t -> t -> IO ()
  _lgamma       :: t -> t -> IO ()
  _log1p        :: t -> t -> IO ()
  _exp          :: t -> t -> IO ()
  _cos          :: t -> t -> IO ()
  _acos         :: t -> t -> IO ()
  _cosh         :: t -> t -> IO ()
  _sin          :: t -> t -> IO ()
  _asin         :: t -> t -> IO ()
  _sinh         :: t -> t -> IO ()
  _tan          :: t -> t -> IO ()
  _atan         :: t -> t -> IO ()
  _atan2        :: t -> t -> t -> IO ()
  _tanh         :: t -> t -> IO ()
  _erf          :: t -> t -> IO ()
  _erfinv       :: t -> t -> IO ()
  _pow          :: t -> t -> HsReal t -> IO ()
  _tpow         :: t -> HsReal t -> t -> IO ()
  _sqrt         :: t -> t -> IO ()
  _rsqrt        :: t -> t -> IO ()
  _ceil         :: t -> t -> IO ()
  _floor        :: t -> t -> IO ()
  _round        :: t -> t -> IO ()
  _trunc        :: t -> t -> IO ()
  _frac         :: t -> t -> IO ()
  _lerp         :: t -> t -> t -> HsReal t -> IO ()

cinv_, cinv :: TensorMathPointwiseFloating t => t -> IO t
cinv_ t = twice t _cinv
cinv  t = withEmpty $ \r -> _cinv r t

sigmoid_, sigmoid :: TensorMathPointwiseFloating t => t -> IO t
sigmoid_ t = twice t _sigmoid
sigmoid  t = withEmpty $ \r -> _sigmoid r t

log_, log :: TensorMathPointwiseFloating t => t -> IO t
log_ t = twice t _log
log  t = withEmpty $ \r -> _log r t

lgamma_, lgamma :: TensorMathPointwiseFloating t => t -> IO t
lgamma_ t = twice t _lgamma
lgamma  t = withEmpty $ \r -> _lgamma r t

log1p_, log1p :: TensorMathPointwiseFloating t => t -> IO t
log1p_ t = twice t _log1p
log1p  t = withEmpty $ \r -> _log1p r t

exp_, exp :: TensorMathPointwiseFloating t => t -> IO t
exp_ t = twice t _exp
exp  t = withEmpty $ \r -> _exp r t

cos_, cos :: TensorMathPointwiseFloating t => t -> IO t
cos_ t = twice t _cos
cos  t = withEmpty $ \r -> _cos r t

acos_, acos :: TensorMathPointwiseFloating t => t -> IO t
acos_ t = twice t _acos
acos  t = withEmpty $ \r -> _acos r t

cosh_, cosh :: TensorMathPointwiseFloating t => t -> IO t
cosh_ t = twice t _cosh
cosh  t = withEmpty $ \r -> _cosh r t

sin_, sin :: TensorMathPointwiseFloating t => t -> IO t
sin_ t = twice t _sin
sin  t = withEmpty $ \r -> _sin r t

asin_, asin :: TensorMathPointwiseFloating t => t -> IO t
asin_ t = twice t _asin
asin  t = withEmpty $ \r -> _asin r t

sinh_, sinh :: TensorMathPointwiseFloating t => t -> IO t
sinh_ t = twice t _sinh
sinh  t = withEmpty $ \r -> _sinh r t

tan_, tan :: TensorMathPointwiseFloating t => t -> IO t
tan_ t = twice t _tan
tan  t = withEmpty $ \r -> _tan r t

atan_, atan :: TensorMathPointwiseFloating t => t -> IO t
atan_ t = twice t _atan
atan  t = withEmpty $ \r -> _atan r t

tanh_, tanh :: TensorMathPointwiseFloating t => t -> IO t
tanh_ t = twice t _tanh
tanh  t = withEmpty $ \r -> _tanh r t

erf_, erf :: TensorMathPointwiseFloating t => t -> IO t
erf_ t = twice t _erf
erf  t = withEmpty $ \r -> _erf r t

erfinv_, erfinv :: TensorMathPointwiseFloating t => t -> IO t
erfinv_ t = twice t _erfinv
erfinv  t = withEmpty $ \r -> _erfinv r t

pow_, pow :: TensorMathPointwiseFloating t => t -> HsReal t -> IO t
pow_ t v = twice t  $ \r t' -> _pow r t' v
pow  t v = withEmpty $ \r -> _pow r t v

tpow_, tpow :: TensorMathPointwiseFloating t => HsReal t -> t -> IO t
tpow_ v t = twice t $ \r t' -> _tpow r v t'
tpow  v t = withEmpty $ \r -> _tpow r v t

sqrt_, sqrt :: TensorMathPointwiseFloating t => t -> IO t
sqrt_ t = twice t _sqrt
sqrt  t = withEmpty $ \r -> _sqrt r t

rsqrt_, rsqrt :: TensorMathPointwiseFloating t => t -> IO t
rsqrt_ t = twice t _rsqrt
rsqrt  t = withEmpty $ \r -> _rsqrt r t

ceil_, ceil :: TensorMathPointwiseFloating t => t -> IO t
ceil_ t = twice t _ceil
ceil  t = withEmpty $ \r -> _ceil r t

floor_, floor :: TensorMathPointwiseFloating t => t -> IO t
floor_ t = twice t _floor
floor  t = withEmpty $ \r -> _floor r t

round_, round :: TensorMathPointwiseFloating t => t -> IO t
round_ t = twice t _round
round  t = withEmpty $ \r -> _round r t

trunc_, trunc :: TensorMathPointwiseFloating t => t -> IO t
trunc_ t = twice t _trunc
trunc  t = withEmpty $ \r -> _trunc r t

frac_, frac :: TensorMathPointwiseFloating t => t -> IO t
frac_ t = twice t _frac
frac  t = withEmpty $ \r -> _frac r t


class CPUTensorMathPointwiseFloating t where
  histc_        :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  bhistc_       :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()


-- clshift_     :: TensorMathPointwise t => t -> t -> IO t
-- crshift_     :: TensorMathPointwise t => t -> t -> IO t
-- cfmod_       :: TensorMathPointwise t => t -> t -> IO t
-- cremainder_  :: TensorMathPointwise t => t -> t -> IO t
-- cmax_        :: TensorMathPointwise t => t -> t -> IO t
-- cmin_        :: TensorMathPointwise t => t -> t -> IO t
-- cmaxValue_   :: TensorMathPointwise t => t -> HsReal t -> IO t
-- cminValue_   :: TensorMathPointwise t => t -> HsReal t -> IO t
-- cbitand_     :: TensorMathPointwise t => t -> t -> IO t
-- cbitor_      :: TensorMathPointwise t => t -> t -> IO t
-- cbitxor_     :: TensorMathPointwise t => t -> t -> IO t
-- addcmul_     :: TensorMathPointwise t => t -> HsReal t -> t -> t -> IO t

-- addcdiv_ :: TensorMathPointwise t => t -> HsReal t -> t -> t -> IO t
-- addcdiv_ t x a b c =

