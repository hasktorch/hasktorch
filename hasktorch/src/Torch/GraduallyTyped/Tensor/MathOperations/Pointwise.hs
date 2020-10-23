{-# LANGUAGE RankNTypes #-}

module Torch.GraduallyTyped.Tensor.MathOperations.Pointwise where

import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Prelude hiding (abs)
import qualified Torch.Internal.Managed.Native as ATen
import System.IO.Unsafe (unsafePerformIO)
import Torch.Internal.Cast (cast3, cast1, cast4, cast2)
import Torch.GraduallyTyped.Scalar (Scalar)

abs ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
abs = undefined

absolute ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
absolute = abs

acos ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
acos = undefined

acosh ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
acosh = undefined

addScalar ::
  forall scalar requiresGradient layout device dataType shape.
  (Scalar scalar) =>
  scalar ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
addScalar scalar tensor = unsafePerformIO $ cast2 ATen.add_ts tensor scalar

addcdiv ::
  forall scalar requiresGradient layout device dataType shape.
  (Scalar scalar) =>
  scalar ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
addcdiv scalar tensor1 tensor2 input = unsafePerformIO $ cast4 ATen.addcdiv_ttts input tensor1 tensor2 scalar

addcmul ::
  forall scalar requiresGradient layout device dataType shape.
  scalar ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
addcmul = undefined

angle ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
angle = undefined

asin ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
asin = undefined

asinh ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
asinh = undefined

atan ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
atan = undefined

atanh ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
atanh = undefined

atan2 ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
atan2 = undefined

bitwiseNot ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
bitwiseNot = undefined

bitwiseAnd ::
  forall requiresGradient layout device dataType shape.
  -- | other
  Tensor requiresGradient layout device dataType shape ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
bitwiseAnd other input = unsafePerformIO $ cast2 ATen.bitwise_and_tt input other

bitwiseOr ::
  forall requiresGradient layout device dataType shape.
  -- | other
  Tensor requiresGradient layout device dataType shape ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
bitwiseOr other input = unsafePerformIO $ cast2 ATen.bitwise_or_tt input other

bitwiseOrScalar ::
  forall scalar requiresGradient layout device dataType shape.
  (Scalar scalar) =>
  scalar ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
bitwiseOrScalar scalar input = unsafePerformIO $ cast2 ATen.bitwise_or_ts input scalar

ceil ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
ceil = unsafePerformIO . cast1 ATen.ceil_t

-- | Clamp all elements in input into the range [ min, max ]
-- and return the result as a new tensor.
clamp ::
  forall min max requiresGradient layout device dataType shape.
  (Scalar min, Scalar max) =>
  min ->
  max ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
clamp min max input = unsafePerformIO $ cast3 ATen.clamp__tss input min max

cos ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
cos = unsafePerformIO . cast1 ATen.cos_t

cosh ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
cosh = unsafePerformIO . cast1 ATen.cosh_t

deg2rad ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
deg2rad = unsafePerformIO . cast1 ATen.deg2rad_t

-- | Element-wise division of the input tensor by the other tensor.
-- Returns the result as a new tensor.
div ::
  forall requiresGradient layout device dataType shape.
  -- | other
  Tensor requiresGradient layout device dataType shape ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
div other input = unsafePerformIO $ cast2 ATen.div_tt input other

divScalar ::
  forall scalar requiresGradient layout device dataType shape.
  (Scalar scalar) =>
  scalar ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
divScalar scalar tensor = unsafePerformIO $ cast2 ATen.div_ts tensor scalar

digamma ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
digamma = undefined

erf ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
erf = undefined

erfc ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
erfc = undefined

erfinv ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
erfinv = undefined

exp ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
exp = undefined

expm1 ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
expm1 = undefined

floor ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
floor = unsafePerformIO . cast1 ATen.min_t

floorDivide ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
floorDivide = undefined

fmod ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
fmod = undefined

frac ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
frac = undefined

lerp ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
lerp = undefined

lgamma ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
lgamma = undefined

log ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
log = undefined

log10 ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
log10 = undefined

log1p ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
log1p = undefined

log2 ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
log2 = undefined

logaddexp ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
logaddexp = undefined

logaddexp2 ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
logaddexp2 = undefined

logicalAnd ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
logicalAnd = undefined

logicalNot ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
logicalNot = undefined

logicalOr ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
logicalOr = undefined

logicalXor ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
logicalXor = undefined

mulScalar ::
  forall scalar requiresGradient layout device dataType shape.
  (Scalar scalar) =>
  scalar ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
mulScalar scalar tensor = unsafePerformIO $ cast2 ATen.mul_ts tensor scalar

mvlgamma ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
mvlgamma = undefined

neg ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
neg = undefined

polygamma ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
polygamma = undefined

pow ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
pow = undefined

rad2deg ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
rad2deg = undefined

reciprocal ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
reciprocal = undefined

remainder ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
remainder = undefined

round ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
round = undefined

rsqrt ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
rsqrt = undefined

sigmoid ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
sigmoid = undefined

sign ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
sign = undefined

sin ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
sin = undefined

sinh ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
sinh = undefined

subScalar ::
  forall scalar requiresGradient layout device dataType shape.
  (Scalar scalar) =>
  scalar ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
subScalar scalar tensor = unsafePerformIO $ cast2 ATen.sub_ts tensor scalar

sqrt ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
sqrt = undefined

square ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
square = undefined

tan ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
tan = undefined

tanh ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
tanh = undefined

trueDivide ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
trueDivide = undefined

trunc ::
  forall requiresGradient layout device dataType shape.
  Tensor requiresGradient layout device dataType shape ->
  Tensor requiresGradient layout device dataType shape
trunc = undefined
