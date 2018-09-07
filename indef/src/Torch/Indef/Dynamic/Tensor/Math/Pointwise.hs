-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Pointwise
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Element-wise functions.
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Pointwise
  ( cross
  , sign_       , sign
  , clamp_      , clamp
  , cadd_       , cadd       , (^+^)
  , csub_       , csub       , (^-^)
  , cmul_       , cmul       , (^*^) , square
  , cdiv_       , cdiv       , (^/^)
  , cpow_       , cpow
  , clshift_    , clshift
  , crshift_    , crshift
  , cfmod_      , cfmod
  , cremainder_ , cremainder
  , cmax_       , cmax
  , cmin_       , cmin
  , cmaxValue_  , cmaxValue
  , cminValue_  , cminValue
  , cbitand_    , cbitand
  , cbitor_     , cbitor
  , cbitxor_    , cbitxor
  , addcmul_    , addcmul
  , addcdiv_    , addcdiv
  ) where

import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor

import qualified Torch.Sig.Tensor.Math.Pointwise as Sig

-- | Replaces all elements in-place with the sign of the elements of the tensor.
sign_ :: Dynamic -> IO ()
sign_ a = _sign a a

-- | Returns a new Tensor with the sign (+/- 1) of the elements of a tensor
sign :: Dynamic -> IO Dynamic
sign t = withEmpty t $ \r -> _sign r t

-- |  returns the cross product of vectors in the specified dimension
--
-- a and b must have the same size, and both @a:size(n)@ and @b:size(n)@ must be 3.
cross :: Dynamic -> Dynamic -> DimVal -> IO Dynamic
cross a b di = withEmpty a $ \r -> _cross r a b di

-- | Clamp all elements, inplace, in the tensor into the range @[min_value, max_value]@. ie:
--
-- @
--         { min_value, if x_i < min_value
--   y_i = { x_i,       if min_value ≤ x_i ≤ max_value
--         { max_value, if x_i > max_value
-- @
clamp_ :: Dynamic -> HsReal -> HsReal -> IO ()
clamp_ t a b = _clamp t t a b

-- | pure version of 'clamp_' returning a new tensor as output.
clamp :: Dynamic -> HsReal -> HsReal -> IO Dynamic
clamp  t a b = withEmpty t $ \r -> _clamp r t a b

-- | Multiply elements of tensor2 by the scalar value and add it to tensor1.
-- The number of elements must match, but sizes do not matter.
--
-- Stores the result in tensor1.
cadd_
  :: Dynamic  -- ^ tensor1
  -> HsReal   -- ^ scale term to multiply againts tensor2
  -> Dynamic  -- ^ tensor2
  -> IO ()
cadd_ t v b = _cadd t t v b

-- | Multiply elements of tensor2 by the scalar value and add it to tensor1.
-- The number of elements must match, but sizes do not matter.
cadd
  :: Dynamic  -- ^ tensor1
  -> HsReal   -- ^ scale term to multiply againts tensor2
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
cadd  t v b = withEmpty t $ \r -> _cadd r t v b

-- | inline alias to 'cadd'
(^+^) :: Dynamic -> Dynamic -> Dynamic
(^+^) a b = unsafePerformIO $ cadd a 1 b
{-# NOINLINE (^+^) #-}

-- | Multiply elements of tensor2 by the scalar value and subtract it from tensor1.
-- The number of elements must match, but sizes do not matter.
--
-- Stores the result in tensor1.
csub_
  :: Dynamic  -- ^ tensor1
  -> HsReal   -- ^ scale term to multiply againts tensor2
  -> Dynamic  -- ^ tensor2
  -> IO ()
csub_ t v b = _csub t t v b

-- | Multiply elements of tensor2 by the scalar value and subtract it from tensor1.
-- The number of elements must match, but sizes do not matter.
csub
  :: Dynamic  -- ^ tensor1
  -> HsReal   -- ^ scale term to multiply againts tensor2
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
csub  t v b = withEmpty t $ \r -> _csub r t v b

-- | inline alias to 'csub'
(^-^) :: Dynamic -> Dynamic -> Dynamic
(^-^) a b = unsafePerformIO $ csub a 1 b
{-# NOINLINE (^-^) #-}

-- | Performs the element-wise multiplication of tensor1 by tensor2. The number of elements must match,
-- but sizes do not matter.
--
-- Stores the result in tensor1.
cmul_
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
cmul_ t1 t2 = _cmul t1 t1 t2

-- | Performs the element-wise multiplication of tensor1 by tensor2. The number of elements must match,
-- but sizes do not matter.
cmul
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
cmul  t1 t2 = withEmpty t1 $ \r -> _cmul r t1 t2

-- | square a tensor.
--
-- FIXME: this is a call to 'cmul' but it might be better to use 'pow'
square :: Dynamic -> IO Dynamic
square t = cmul t t

-- | inline alias to 'cmul'
(^*^) :: Dynamic -> Dynamic -> Dynamic
(^*^) a b = unsafePerformIO $ cmul a b
{-# NOINLINE (^*^) #-}

-- | Performs the element-wise division of tensor1 by tensor2. The number of elements must match,
-- but sizes do not matter.
--
-- Stores the result in tensor1.
cdiv_
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
cdiv_ t1 t2 = _cdiv t1 t1 t2

-- | Performs the element-wise division of tensor1 by tensor2. The number of elements must match,
-- but sizes do not matter.
cdiv
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
cdiv  t1 t2 = withEmpty t1 $ \r -> _cdiv r t1 t2

-- | inline alias to 'cdiv'
(^/^) :: Dynamic -> Dynamic -> Dynamic
(^/^) a b = unsafePerformIO $ cdiv a b
{-# NOINLINE (^/^) #-}

-- | Element-wise power operation, taking the elements of tensor1 to the powers given by elements
-- of tensor2. The number of elements must match, but sizes do not matter.
--
-- Stores the result in tensor1.
cpow_
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
cpow_ = _mkInplaceFunction _cpow

-- | Element-wise power operation, taking the elements of tensor1 to the powers given by elements
-- of tensor2. The number of elements must match, but sizes do not matter.
cpow
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
cpow  = _mkNewFunction _cpow

-- | Element-wise, inplace, bitwise left shift. Mutates the first argument.
clshift_ :: Dynamic -> Dynamic -> IO ()
clshift_ = _mkInplaceFunction _clshift

-- | Element-wise bitwise left shift.
clshift :: Dynamic -> Dynamic -> IO Dynamic
clshift  = _mkNewFunction _clshift

-- | Element-wise, inplace, bitwise right shift. Mutates the first argument.
crshift_ :: Dynamic -> Dynamic -> IO ()
crshift_ = _mkInplaceFunction _crshift

-- | Element-wise bitwise right shift.
crshift :: Dynamic -> Dynamic -> IO Dynamic
crshift  = _mkNewFunction _crshift

-- | Computes the element-wise remainder of the division (rounded towards zero)
-- of tensor1 by tensor2. The number of elements must match, but sizes do not matter.
--
-- Stores the result in tensor1.
cfmod_
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
cfmod_ = _mkInplaceFunction _cfmod

-- | Computes the element-wise remainder of the division (rounded towards zero)
-- of tensor1 by tensor2. The number of elements must match, but sizes do not matter.
cfmod
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
cfmod  = _mkNewFunction _cfmod

-- | Computes element-wise remainder of the division (rounded to nearest) of
-- tensor1 by tensor2. The number of elements must match, but sizes do not matter.
--
-- Stores the result in tensor1.
cremainder_
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
cremainder_ = _mkInplaceFunction     _cremainder

-- | Computes element-wise remainder of the division (rounded to nearest) of
-- tensor1 by tensor2. The number of elements must match, but sizes do not matter.
cremainder
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
cremainder  = _mkNewFunction _cremainder

-- | Element-wise, inplace, max function -- storing the higher value. Mutates the first argument.
cmax_ :: Dynamic -> Dynamic -> IO ()
cmax_ = _mkInplaceFunction     _cmax

-- | Element-wise max function -- keeping the higher value.
cmax :: Dynamic -> Dynamic -> IO Dynamic
cmax  = _mkNewFunction _cmax

-- | Element-wise, inplace, min function -- storing the lower value. Mutates the first argument.
cmin_ :: Dynamic -> Dynamic -> IO ()
cmin_ = _mkInplaceFunction     _cmin

-- | Element-wise min function -- keeping the lower value.
cmin :: Dynamic -> Dynamic -> IO Dynamic
cmin  = _mkNewFunction _cmin

-- | Store the maximum of the tensor and a scalar in the input tensor on an element-by-element basis.
cmaxValue_ :: Dynamic -> HsReal -> IO ()
cmaxValue_ t v = _cmaxValue t t v

-- | Store the maximum of the tensor and a scalar in a new tensor on an element-by-element basis.
cmaxValue :: Dynamic -> HsReal -> IO Dynamic
cmaxValue  t v = withEmpty t $ \r -> _cmaxValue r t v

-- | Store the minimum of the tensor and a scalar in the input tensor on an element-by-element basis.
cminValue_ :: Dynamic -> HsReal -> IO ()
cminValue_ t v = _cminValue t t v

-- | Store the minimum of the tensor and a scalar in a new tensor on an element-by-element basis.
cminValue :: Dynamic -> HsReal -> IO Dynamic
cminValue  t v = withEmpty t $ \r -> _cminValue r t v

-- | Element-wise, inplace, bitwise @and@. Mutates the first argument.
cbitand_ :: Dynamic -> Dynamic -> IO ()
cbitand_ = _mkInplaceFunction     _cbitand

-- | Element-wise, bitwise @and@
cbitand :: Dynamic -> Dynamic -> IO Dynamic
cbitand  = _mkNewFunction _cbitand

-- | Element-wise, inplace, bitwise @or@. Mutates the first argument.
cbitor_ :: Dynamic -> Dynamic -> IO ()
cbitor_  = _mkInplaceFunction     _cbitor

-- | Element-wise, bitwise @or@
cbitor :: Dynamic -> Dynamic -> IO Dynamic
cbitor   = _mkNewFunction _cbitor

-- | Element-wise, inplace, bitwise @xor@. Mutates the first argument.
cbitxor_ :: Dynamic -> Dynamic -> IO ()
cbitxor_ = _mkInplaceFunction     _cbitxor

-- | Element-wise, bitwise @xor@
cbitxor :: Dynamic -> Dynamic -> IO Dynamic
cbitxor  = _mkNewFunction _cbitxor

-- | Performs the element-wise multiplication of tensor1 by tensor2, multiplies
-- the result by the scalar value and adds it to tensor0, which it mutates inplace.
--
-- The number of elements must match, but sizes do not matter.
addcmul_
  :: Dynamic  -- ^ tensor0
  -> HsReal   -- ^ scale term
  -> Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
addcmul_ a v b c = _addcmul a a v b c

-- | Performs the element-wise multiplication of tensor1 by tensor2, multiplies
-- the result by the scalar value and adds it to tensor0.
--
-- The number of elements must match, but sizes do not matter.
addcmul
  :: Dynamic  -- ^ tensor0
  -> HsReal   -- ^ scale term
  -> Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
addcmul  a v b c = withEmpty a $ \r -> _addcmul r a v b c


-- | Performs the element-wise multiplication of tensor1 by tensor2, multiplies
-- the result by the scalar value and adds it to tensor0, which it mutates inplace.
--
-- The number of elements must match, but sizes do not matter.
addcdiv_
  :: Dynamic  -- ^ tensor0
  -> HsReal   -- ^ scale term
  -> Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
addcdiv_ a v b c = _addcdiv a a v b c

-- | Performs the element-wise division of tensor1 by tensor2, multiplies
-- the result by the scalar value and adds it to tensor0.
--
-- The number of elements must match, but sizes do not matter.
addcdiv
  :: Dynamic  -- ^ tensor0
  -> HsReal   -- ^ scale term
  -> Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO Dynamic
addcdiv  a v b c = withEmpty a $ \r -> _addcdiv r a v b c

-- ========================================================================= --
-- raw C-level calls

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

-- ========================================================================= --
-- helper functions

_mkNewFunction :: (Dynamic -> Dynamic -> Dynamic -> IO ()) -> Dynamic -> Dynamic -> IO Dynamic
_mkNewFunction op t1 t2 = withEmpty t1 $ \r -> op r t1 t2

_mkInplaceFunction :: (Dynamic -> Dynamic -> Dynamic -> IO ()) -> Dynamic -> Dynamic -> IO ()
_mkInplaceFunction op t1 t2 = op t1 t1 t2

