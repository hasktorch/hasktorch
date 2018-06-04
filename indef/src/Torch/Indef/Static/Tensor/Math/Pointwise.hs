{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.Tensor.Math.Pointwise where

import System.IO.Unsafe
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise as Dynamic

_sign r t = Dynamic._sign (asDynamic r) (asDynamic t)
_cross ret a b d = Dynamic._cross (asDynamic ret) (asDynamic a) (asDynamic b) d
_clamp ret a mn mx = Dynamic._clamp (asDynamic ret) (asDynamic a) mn mx
_cadd ret a v b = Dynamic._cadd (asDynamic ret) (asDynamic a) v (asDynamic b)
_csub ret a v b = Dynamic._csub (asDynamic ret) (asDynamic a) v (asDynamic b)
_cmul ret a b = Dynamic._cmul (asDynamic ret) (asDynamic a) (asDynamic b)
_cpow ret a b = Dynamic._cpow (asDynamic ret) (asDynamic a) (asDynamic b)
_cdiv ret a b = Dynamic._cdiv (asDynamic ret) (asDynamic a) (asDynamic b)
_clshift ret a b = Dynamic._clshift (asDynamic ret) (asDynamic a) (asDynamic b)
_crshift ret a b = Dynamic._crshift (asDynamic ret) (asDynamic a) (asDynamic b)
_cfmod ret a b = Dynamic._cfmod (asDynamic ret) (asDynamic a) (asDynamic b)
_cremainder ret a b = Dynamic._cremainder (asDynamic ret) (asDynamic a) (asDynamic b)
_cmax ret a b = Dynamic._cmax (asDynamic ret) (asDynamic a) (asDynamic b)
_cmin ret a b = Dynamic._cmin (asDynamic ret) (asDynamic a) (asDynamic b)
_cmaxValue ret a v = Dynamic._cmaxValue (asDynamic ret) (asDynamic a) v
_cminValue ret a v = Dynamic._cminValue (asDynamic ret) (asDynamic a) v
_cbitand ret a b = Dynamic._cbitand (asDynamic ret) (asDynamic a) (asDynamic b)
_cbitor ret a b = Dynamic._cbitor (asDynamic ret) (asDynamic a) (asDynamic b)
_cbitxor ret a b = Dynamic._cbitxor (asDynamic ret) (asDynamic a) (asDynamic b)
_addcmul ret a v b c = Dynamic._addcmul (asDynamic ret) (asDynamic a) v (asDynamic b) (asDynamic c)
_addcdiv ret a v b c = Dynamic._addcdiv (asDynamic ret) (asDynamic a) v (asDynamic b) (asDynamic c)

sign_, sign :: (Dimensions d) => Tensor d -> IO (Tensor d)
sign_ t = withInplace t _sign
sign  t = withEmpty $ \r -> _sign r t

clamp_, clamp :: (Dimensions d) => Tensor d -> HsReal -> HsReal -> IO (Tensor d)
clamp_ t a b = withInplace t $ \r' t' -> _clamp r' t' a b
clamp  t a b = withEmpty $ \r -> _clamp r t a b

cadd_, cadd :: (Dimensions d) => Tensor d -> HsReal -> Tensor d -> IO (Tensor d)
cadd_ t v b = withInplace t $ \r' t' -> _cadd r' t' v b
cadd  t v b = withEmpty $ \r -> _cadd r t v b
(^+^) :: (Dimensions d) => Tensor d -> Tensor d -> Tensor d
(^+^) a b = unsafePerformIO $ cadd a 1 b
{-# NOINLINE (^+^) #-}

csub_, csub :: (Dimensions d) => Tensor d -> HsReal -> Tensor d -> IO (Tensor d)
csub_ t v b = withInplace t $ \r' t' -> _csub r' t' v b
csub  t v b = withEmpty $ \r -> _csub r t v b
(^-^) :: (Dimensions d) => Tensor d -> Tensor d -> Tensor d
(^-^) a b = unsafePerformIO $ csub a 1 b
{-# NOINLINE (^-^) #-}

cmul_, cmul :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cmul_ t1 t2 = withInplace t1 $ \r' t1' -> _cmul r' t1' t2
cmul  t1 t2 = withEmpty $ \r -> _cmul r t1 t2
square :: (Dimensions d) => Tensor d -> IO (Tensor d)
square t = cmul t t
(^*^) :: (Dimensions d) => Tensor d -> Tensor d -> Tensor d
(^*^) a b = unsafePerformIO $ cmul a b
{-# NOINLINE (^*^) #-}

cdiv_, cdiv :: Dimensions d => Tensor d -> Tensor d -> IO (Tensor d)
cdiv_ t1 t2 = withInplace t1 $ \r' t1' -> _cdiv r' t1' t2
cdiv  t1 t2 = withEmpty $ \r -> _cdiv r t1 t2
(^/^) :: (Dimensions d) => Tensor d -> Tensor d -> Tensor d
(^/^) a b = unsafePerformIO $ cdiv a b
{-# NOINLINE (^/^) #-}

cpow_, cpow  :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cpow_ t1 t2 = withInplace t1 $ \r' t1' -> _cpow r' t1' t2
cpow  t1 t2 = withEmpty $ \r -> _cpow r t1 t2

clshift_, clshift, crshift_, crshift, cfmod, cfmod_, cremainder, cremainder_
  :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
clshift_ a b = _clshift a a b >> pure (asStatic (asDynamic a))
clshift  a b = withEmpty $ \r -> _clshift r a b
crshift_ a b = _crshift a a b >> pure (asStatic (asDynamic a))
crshift  a b = withEmpty $ \r -> _crshift r a b
cfmod_ a b = _cfmod a a b >> pure (asStatic (asDynamic a))
cfmod  a b = withEmpty $ \r -> _cfmod r a b
cremainder_ a b = _cremainder a a b >> pure (asStatic (asDynamic a))
cremainder  a b = withEmpty $ \r -> _cremainder r a b

cmax, cmax_, cmin, cmin_, cbitand, cbitand_, cbitor, cbitor_, cbitxor, cbitxor_
  :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cmax_ a b = _cmax a a b >> pure (asStatic (asDynamic a))
cmax  a b = withEmpty $ \r -> _cmax r a b
cmin_ a b = _cmin a a b >> pure (asStatic (asDynamic a))
cmin  a b = withEmpty $ \r -> _cmin r a b
cbitand_ a b = _cbitand a a b >> pure (asStatic (asDynamic a))
cbitand  a b = withEmpty $ \r -> _cbitand r a b
cbitor_ a b = _cbitor a a b >> pure (asStatic (asDynamic a))
cbitor  a b = withEmpty $ \r -> _cbitor r a b
cbitxor_ a b = _cbitxor a a b >> pure (asStatic (asDynamic a))
cbitxor  a b = withEmpty $ \r -> _cbitxor r a b

addcmul, addcmul_, addcdiv, addcdiv_
  :: (Dimensions d) => Tensor d -> HsReal -> Tensor d -> Tensor d -> IO (Tensor d)
addcmul  a v b c = withEmpty $ \r -> _addcmul r a v b c
addcmul_ a v b c = _addcmul a a v b c >> pure (asStatic (asDynamic a))
addcdiv  a v b c = withEmpty $ \r -> _addcdiv r a v b c
addcdiv_ a v b c = _addcdiv a a v b c >> pure (asStatic (asDynamic a))



