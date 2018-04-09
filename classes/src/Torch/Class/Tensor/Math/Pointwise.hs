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

class (Num (HsReal t), Tensor t) => TensorMathPointwise t where
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

class TensorMathPointwiseSigned t where
  _neg :: t -> t -> IO ()
  _abs :: t -> t -> IO ()

class TensorMathPointwiseFloating t where
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

class CPUTensorMathPointwiseFloating t where
  histc_        :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  bhistc_       :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()


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
(^+^) :: TensorMathPointwise t => t -> t -> t
(^+^) a b = unsafePerformIO $ cadd a 1 b
{-# NOINLINE (^+^) #-}

csub_, csub :: TensorMathPointwise t => t -> HsReal t -> t -> IO t
csub_ t v b = t `twice` (\r' t' -> _csub r' t' v b)
csub  t v b = withEmpty $ \r -> _csub r t v b
(^-^) :: TensorMathPointwise t => t -> t -> t
(^-^) a b = unsafePerformIO $ csub a 1 b
{-# NOINLINE (^-^) #-}

cmul_, cmul :: TensorMathPointwise t => t -> t -> IO t
cmul_ t1 t2 = t1 `twice` (\r' t1' -> _cmul r' t1' t2)
cmul  t1 t2 = withEmpty $ \r -> _cmul r t1 t2
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

