{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Pointwise where

import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Dimensions
import Torch.Indef.Dynamic.Tensor

import qualified Torch.Sig.Tensor.Math.Pointwise as Sig

_sign :: Dynamic -> Dynamic -> IO ()
_sign r t = with2DynamicState r t Sig.c_sign

_cross :: Dynamic -> Dynamic -> Dynamic -> DimVal -> IO ()
_cross t0 t1 t2 i0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_cross s' t0' t1' t2' (fromIntegral i0)

_clamp :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
_clamp r t v0 v1 = with2DynamicState r t $ shuffle3'2 Sig.c_clamp (hs2cReal v0) (hs2cReal v1)

_cmax :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cmax r t0 t1 = with3DynamicState r t0 t1 $ \s' r' t0' t1' ->  Sig.c_cmax s' r' t0' t1'

_cmin :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cmin r t0 t1 = with3DynamicState r t0 t1 $ \s' r' t0' t1' ->  Sig.c_cmin s' r' t0' t1'

_cmaxValue :: Dynamic -> Dynamic -> HsReal -> IO ()
_cmaxValue r t v = with2DynamicState r t $ shuffle3 Sig.c_cmaxValue (hs2cReal v)

_cminValue :: Dynamic -> Dynamic -> HsReal -> IO ()
_cminValue r t v = with2DynamicState r t $ shuffle3 Sig.c_cminValue (hs2cReal v)

_addcmul :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
_addcmul t0 t1 v t2 t3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_addcmul s' t0' t1' (hs2cReal v) t2' t3'

_addcdiv :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
_addcdiv t0 t1 v t2 t3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_ t2' t3' ->
      Sig.c_addcdiv s' t0' t1' (hs2cReal v) t2' t3'

_cadd :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
_cadd t0 t1 v t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_cadd s' t0' t1' (hs2cReal v) t2'

_csub :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
_csub t0 t1 v t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' -> Sig.c_csub s' t0' t1' (hs2cReal v) t2'

_cmul :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cmul r t0 t1 = with3DynamicState r t0 t1 Sig.c_cmul

_cpow :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cpow r t0 t1 = with3DynamicState r t0 t1 Sig.c_cpow

_cdiv :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cdiv r t0 t1 = with3DynamicState r t0 t1 Sig.c_cdiv

_clshift :: Dynamic -> Dynamic -> Dynamic -> IO ()
_clshift r t0 t1 = with3DynamicState r t0 t1 Sig.c_clshift

_crshift :: Dynamic -> Dynamic -> Dynamic -> IO ()
_crshift r t0 t1 = with3DynamicState r t0 t1 Sig.c_crshift

_cfmod :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cfmod r t0 t1 = with3DynamicState r t0 t1 Sig.c_cfmod

_cremainder :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cremainder r t0 t1 = with3DynamicState r t0 t1 Sig.c_cremainder

_cbitand :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cbitand r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitand

_cbitor :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cbitor r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitor

_cbitxor :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cbitxor r t0 t1 = with3DynamicState r t0 t1 Sig.c_cbitxor

sign_, sign :: Dynamic -> IO Dynamic
sign_  = (`twice` _sign)
sign t = withEmpty t $ \r -> _sign r t

cross :: Dynamic -> Dynamic -> DimVal -> IO Dynamic
cross a b di = withEmpty a $ \r -> _cross r a b di

clamp_, clamp :: Dynamic -> HsReal -> HsReal -> IO Dynamic
clamp_ t a b = t `twice` (\r' t' -> _clamp r' t' a b)
clamp  t a b = withEmpty t $ \r -> _clamp r t a b

cadd_, cadd :: Dynamic -> HsReal -> Dynamic -> IO Dynamic
cadd_ t v b = t `twice` (\r' t' -> _cadd r' t' v b)
cadd  t v b = withEmpty t $ \r -> _cadd r t v b
(^+^) :: Dynamic -> Dynamic -> Dynamic
(^+^) a b = unsafePerformIO $ cadd a 1 b
{-# NOINLINE (^+^) #-}

csub_, csub :: Dynamic -> HsReal -> Dynamic -> IO Dynamic
csub_ t v b = t `twice` (\r' t' -> _csub r' t' v b)
csub  t v b = withEmpty t $ \r -> _csub r t v b
(^-^) :: Dynamic -> Dynamic -> Dynamic
(^-^) a b = unsafePerformIO $ csub a 1 b
{-# NOINLINE (^-^) #-}

cmul_, cmul :: Dynamic -> Dynamic -> IO Dynamic
cmul_ t1 t2 = t1 `twice` (\r' t1' -> _cmul r' t1' t2)
cmul  t1 t2 = withEmpty t1 $ \r -> _cmul r t1 t2
square :: Dynamic -> IO Dynamic
square t = cmul t t
(^*^) :: Dynamic -> Dynamic -> Dynamic
(^*^) a b = unsafePerformIO $ cmul a b
{-# NOINLINE (^*^) #-}

cdiv_, cdiv :: Dynamic -> Dynamic -> IO Dynamic
cdiv_ t1 t2 = t1 `twice` (\r' t1' -> _cdiv r' t1' t2)
cdiv  t1 t2 = withEmpty t1 $ \r -> _cdiv r t1 t2
(^/^) :: Dynamic -> Dynamic -> Dynamic
(^/^) a b = unsafePerformIO $ cdiv a b
{-# NOINLINE (^/^) #-}

_mkNewFunction, _mkInplaceFunction
  :: (Dynamic -> Dynamic -> Dynamic -> IO ()) -> Dynamic -> Dynamic -> IO Dynamic
_mkNewFunction     op t1 t2 = t1 `twice` (\r' t1' -> op r' t1' t2)
_mkInplaceFunction op t1 t2 = withEmpty t1 $ \r -> op r t1 t2


cpow_, cpow, clshift_, clshift, crshift_, crshift
  :: Dynamic -> Dynamic -> IO Dynamic
cpow_ = _mkNewFunction     _cpow
cpow  = _mkInplaceFunction _cpow
clshift_ = _mkNewFunction     _clshift
clshift  = _mkInplaceFunction _clshift
crshift_ = _mkNewFunction     _crshift
crshift  = _mkInplaceFunction _crshift

cfmod_, cfmod, cremainder_, cremainder, cmax_, cmax, cmin_, cmin
  :: Dynamic -> Dynamic -> IO Dynamic
cfmod_ = _mkNewFunction     _cfmod
cfmod  = _mkInplaceFunction _cfmod
cremainder_ = _mkNewFunction     _cremainder
cremainder  = _mkInplaceFunction _cremainder
cmax_ = _mkNewFunction     _cmax
cmax  = _mkInplaceFunction _cmax
cmin_ = _mkNewFunction     _cmin
cmin  = _mkInplaceFunction _cmin

cmaxValue_, cmaxValue :: Dynamic -> HsReal -> IO Dynamic
cmaxValue_ t v = t `twice` (\r' t' -> _cmaxValue r' t' v)
cmaxValue  t v = withEmpty t $ \r -> _cmaxValue r t v

cminValue_, cminValue :: Dynamic -> HsReal -> IO Dynamic
cminValue_ t v = t `twice` (\r' t' -> _cminValue r' t' v)
cminValue  t v = withEmpty t $ \r -> _cminValue r t v

cbitand_, cbitand, cbitor_, cbitor, cbitxor_, cbitxor 
  :: Dynamic -> Dynamic -> IO Dynamic
cbitand_ = _mkNewFunction     _cbitand
cbitand  = _mkInplaceFunction _cbitand
cbitor_  = _mkNewFunction     _cbitor
cbitor   = _mkInplaceFunction _cbitor
cbitxor_ = _mkNewFunction     _cbitxor
cbitxor  = _mkInplaceFunction _cbitxor


addcmul_, addcmul :: Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
addcmul_ a v b c = a `twice` (\r' a' -> _addcmul r' a' v b c)
addcmul  a v b c = withEmpty a $ \r -> _addcmul r a v b c

addcdiv_, addcdiv :: Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
addcdiv_ a v b c = a `twice` (\r' a' -> _addcdiv r' a' v b c)
addcdiv  a v b c = withEmpty a $ \r -> _addcdiv r a v b c

