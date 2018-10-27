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

import Debug.Trace
import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor

import qualified Torch.Sig.Tensor.Math.Pointwise as Sig

-- | Replaces all elements in-place with the sign of the elements of the tensor.
sign_ :: Dynamic -> IO ()
sign_ a = _sign a a

-- | Returns a new Tensor with the sign (+/- 1) of the elements of a tensor
sign :: Dynamic -> Dynamic
sign t = unsafeDupablePerformIO $ do
  let r = empty
  _sign r t
  pure r
{-# NOINLINE sign #-}

-- |  returns the cross product of vectors in the specified dimension
--
-- a and b must have the same size, and both @a:size(n)@ and @b:size(n)@ must be 3.
cross
  :: Dynamic    -- ^ tensor a
  -> Dynamic    -- ^ tensor b (same size as tensor a in dimension below)
  -> Word       -- ^ dimension to operate over
  -> Dynamic    -- ^ new return tensor
cross a b di = unsafeDupablePerformIO $ do
  let r = empty
  _cross r a b di
  pure r
{-# NOINLINE cross #-}

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
clamp :: Dynamic -> HsReal -> HsReal -> Dynamic
clamp  t a b = unsafeDupablePerformIO $ do
  let r = new' (getSomeDims t)
  _clamp r t a b
  pure r
{-# NOINLINE clamp #-}

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
  -> Dynamic
cadd t v b = unsafeDupablePerformIO $ do
  let r = new' (getSomeDims t)
  _cadd r t v b
  pure r
{-# NOINLINE cadd #-}

-- | inline alias to 'cadd'
(^+^) :: Dynamic -> Dynamic -> Dynamic
(^+^) a b = cadd a 1 b

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
  -> Dynamic
csub t v b = unsafeDupablePerformIO $ do
  let r = new' (getSomeDims t)
  _csub r t v b
  pure r
{-# NOINLINE csub #-}

-- | inline alias to 'csub'
(^-^) :: Dynamic -> Dynamic -> Dynamic
(^-^) a b = csub a 1 b

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
  -> Dynamic
cmul t1 t2 = unsafeDupablePerformIO $ do
  let r = new' (getSomeDims t1)
  _cmul r t1 t2
  pure r
{-# NOINLINE cmul #-}

-- | square a tensor.
--
-- FIXME: this is a call to 'cmul' but it might be better to use 'pow'
square :: Dynamic -> Dynamic
square t = cmul t t

-- | inline alias to 'cmul'
(^*^) :: Dynamic -> Dynamic -> Dynamic
(^*^) = cmul

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
  -> Dynamic
cdiv  t1 t2 = unsafeDupablePerformIO $ do
  let r = new' (getSomeDims t1)
  _cdiv r t1 t2
  pure r
{-# NOINLINE cdiv #-}

-- | inline alias to 'cdiv'
(^/^) :: Dynamic -> Dynamic -> Dynamic
(^/^) = cdiv

-- | Element-wise power operation, taking the elements of tensor1 to the powers given by elements
-- of tensor2. The number of elements must match, but sizes do not matter.
--
-- Stores the result in tensor1.
cpow_
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
cpow_ t = _cpow t t

-- | Element-wise power operation, taking the elements of tensor1 to the powers given by elements
-- of tensor2. The number of elements must match, but sizes do not matter.
cpow
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> Dynamic
cpow = _mkNewFunction _cpow

-- | Element-wise, inplace, bitwise left shift. Mutates the first argument.
clshift_ :: Dynamic -> Dynamic -> IO ()
clshift_ t = _clshift t t

-- | Element-wise bitwise left shift.
clshift :: Dynamic -> Dynamic -> Dynamic
clshift = _mkNewFunction _clshift

-- | Element-wise, inplace, bitwise right shift. Mutates the first argument.
crshift_ :: Dynamic -> Dynamic -> IO ()
crshift_ t = _crshift t t

-- | Element-wise bitwise right shift.
crshift :: Dynamic -> Dynamic -> Dynamic
crshift = _mkNewFunction _crshift

-- | Computes the element-wise remainder of the division (rounded towards zero)
-- of tensor1 by tensor2. The number of elements must match, but sizes do not matter.
--
-- Stores the result in tensor1.
cfmod_
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
cfmod_ t = _cfmod t t

-- | Computes the element-wise remainder of the division (rounded towards zero)
-- of tensor1 by tensor2. The number of elements must match, but sizes do not matter.
cfmod
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> Dynamic
cfmod = _mkNewFunction _cfmod

-- | Computes element-wise remainder of the division (rounded to nearest) of
-- tensor1 by tensor2. The number of elements must match, but sizes do not matter.
--
-- Stores the result in tensor1.
cremainder_
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> IO ()
cremainder_ t = _cremainder t t

-- | Computes element-wise remainder of the division (rounded to nearest) of
-- tensor1 by tensor2. The number of elements must match, but sizes do not matter.
cremainder
  :: Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> Dynamic
cremainder = _mkNewFunction _cremainder

-- | Element-wise, inplace, max function -- storing the higher value. Mutates the first argument.
cmax_ :: Dynamic -> Dynamic -> IO ()
cmax_ t = _cmax t t

-- | Element-wise max function -- keeping the higher value.
cmax :: Dynamic -> Dynamic -> Dynamic
cmax = _mkNewFunction _cmax

-- | Element-wise, inplace, min function -- storing the lower value. Mutates the first argument.
cmin_ :: Dynamic -> Dynamic -> IO ()
cmin_ t =  _cmin t t

-- | Element-wise min function -- keeping the lower value.
cmin :: Dynamic -> Dynamic -> Dynamic
cmin = _mkNewFunction _cmin

-- | Store the maximum of the tensor and a scalar in the input tensor on an element-by-element basis.
cmaxValue_ :: Dynamic -> HsReal -> IO ()
cmaxValue_ t v = _cmaxValue t t v

-- | Store the maximum of the tensor and a scalar in a new tensor on an element-by-element basis.
cmaxValue :: Dynamic -> HsReal -> Dynamic
cmaxValue t v = unsafeDupablePerformIO $ do
  let r = empty
  _cmaxValue r t v
  pure r
{-# NOINLINE cmaxValue #-}

-- | Store the minimum of the tensor and a scalar in the input tensor on an element-by-element basis.
cminValue_ :: Dynamic -> HsReal -> IO ()
cminValue_ t v = _cminValue t t v

-- | Store the minimum of the tensor and a scalar in a new tensor on an element-by-element basis.
cminValue :: Dynamic -> HsReal -> Dynamic
cminValue  t v = unsafeDupablePerformIO $ do
  let r = empty
  _cminValue r t v
  pure r
{-# NOINLINE cminValue #-}

-- | Element-wise, inplace, bitwise @and@. Mutates the first argument.
cbitand_ :: Dynamic -> Dynamic -> IO ()
cbitand_ t = _cbitand t t

-- | Element-wise, bitwise @and@
cbitand :: Dynamic -> Dynamic -> Dynamic
cbitand = _mkNewFunction _cbitand

-- | Element-wise, inplace, bitwise @or@. Mutates the first argument.
cbitor_ :: Dynamic -> Dynamic -> IO ()
cbitor_ t = _cbitor t t

-- | Element-wise, bitwise @or@
cbitor :: Dynamic -> Dynamic -> Dynamic
cbitor = _mkNewFunction _cbitor

-- | Element-wise, inplace, bitwise @xor@. Mutates the first argument.
cbitxor_ :: Dynamic -> Dynamic -> IO ()
cbitxor_ t =  _cbitxor t t

-- | Element-wise, bitwise @xor@
cbitxor :: Dynamic -> Dynamic -> Dynamic
cbitxor = _mkNewFunction _cbitxor

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
  -> Dynamic
addcmul a v b c = unsafeDupablePerformIO $ do
  let r = empty
  _addcmul r a v b c
  pure r
{-# NOINLINE addcmul #-}


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
addcdiv_ a = _addcdiv a a

-- | Performs the element-wise division of tensor1 by tensor2, multiplies
-- the result by the scalar value and adds it to tensor0.
--
-- The number of elements must match, but sizes do not matter.
addcdiv
  :: Dynamic  -- ^ tensor0
  -> HsReal   -- ^ scale term
  -> Dynamic  -- ^ tensor1
  -> Dynamic  -- ^ tensor2
  -> Dynamic
addcdiv a v b c = unsafeDupablePerformIO $ do
  let r = empty
  _addcdiv r a v b c
  pure r
{-# NOINLINE addcdiv #-}

-- ========================================================================= --
-- raw C-level calls

_sign :: Dynamic -> Dynamic -> IO ()
_sign r t = withLift $ Sig.c_sign
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t

_cross :: Dynamic -> Dynamic -> Dynamic -> Word -> IO ()
_cross t0 t1 t2 i0 = withLift $ Sig.c_cross
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> managedTensor t2
  <*> pure (fromIntegral i0)

_clamp :: Dynamic -> Dynamic -> HsReal -> HsReal -> IO ()
_clamp r t v0 v1 = withLift $ Sig.c_clamp
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t
  <*> pure (hs2cReal v0)
  <*> pure (hs2cReal v1)

_cmax :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cmax r t0 t1 = withLift $ Sig.c_cmax
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cmin :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cmin r t0 t1 = withLift $ Sig.c_cmin
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cmaxValue :: Dynamic -> Dynamic -> HsReal -> IO ()
_cmaxValue r t v = withLift $ Sig.c_cmaxValue
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t
  <*> pure (hs2cReal v)

_cminValue :: Dynamic -> Dynamic -> HsReal -> IO ()
_cminValue r t v = withLift $ Sig.c_cminValue
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t
  <*> pure (hs2cReal v)

_addcmul :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
_addcmul t0 t1 v t2 t3 = withLift $ Sig.c_addcmul
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (hs2cReal v)
  <*> managedTensor t2
  <*> managedTensor t3

_addcdiv :: Dynamic -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
_addcdiv t0 t1 v t2 t3 = withLift $ Sig.c_addcdiv
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (hs2cReal v)
  <*> managedTensor t2
  <*> managedTensor t3

_cadd :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
_cadd t0 t1 v t2 = withLift $ Sig.c_cadd
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (hs2cReal v)
  <*> managedTensor t2

_csub :: Dynamic -> Dynamic -> HsReal -> Dynamic -> IO ()
_csub t0 t1 v t2 = withLift $ Sig.c_csub
  <$> managedState
  <*> managedTensor t0
  <*> managedTensor t1
  <*> pure (hs2cReal v)
  <*> managedTensor t2

_cmul :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cmul r t0 t1 = withLift $ Sig.c_cmul
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cpow :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cpow r t0 t1 = withLift $ Sig.c_cpow
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cdiv :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cdiv r t0 t1 = withLift $ Sig.c_cdiv
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_clshift :: Dynamic -> Dynamic -> Dynamic -> IO ()
_clshift r t0 t1 = withLift $ Sig.c_clshift
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_crshift :: Dynamic -> Dynamic -> Dynamic -> IO ()
_crshift r t0 t1 = withLift $ Sig.c_crshift
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cfmod :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cfmod r t0 t1 = withLift $ Sig.c_cfmod
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cremainder :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cremainder r t0 t1 = withLift $ Sig.c_cremainder
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cbitand :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cbitand r t0 t1 = withLift $ Sig.c_cbitand
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cbitor :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cbitor r t0 t1 = withLift $ Sig.c_cbitor
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

_cbitxor :: Dynamic -> Dynamic -> Dynamic -> IO ()
_cbitxor r t0 t1 = withLift $ Sig.c_cbitxor
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t0
  <*> managedTensor t1

-- ========================================================================= --
-- helper functions

_mkNewFunction :: (Dynamic -> Dynamic -> Dynamic -> IO ()) -> Dynamic -> Dynamic -> Dynamic
_mkNewFunction op t1 t2 = unsafeDupablePerformIO $ do
  let r = empty
  op r t1 t2
  pure r

