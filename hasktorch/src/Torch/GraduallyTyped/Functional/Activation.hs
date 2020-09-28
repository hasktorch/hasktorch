{-# LANGUAGE RankNTypes #-}

module Torch.GraduallyTyped.Functional.Activation where

import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Tensor
import Torch.Internal.Cast (cast1, cast2, cast3)
import qualified Torch.Internal.Managed.Native as ATen
import Torch.Scalar (Scalar)

-- | Thresholds each element of the input Tensor.
threshold ::
  forall threshold value.
  (Scalar threshold, Scalar value) =>
  -- | threshold
  threshold ->
  -- | value
  value ->
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
threshold threshold value tensor =
  unsafePerformIO $ cast3 ATen.threshold_tss tensor threshold value

-- | Applies the rectified linear unit function element-wise, that is,
-- \[
-- \text{ReLU}(x) = max(0, x).
-- \]
relu ::
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
relu = unsafePerformIO . cast1 ATen.relu_t

-- | Applies the HardTanh function element-wise.
hardtanh ::
  forall minValue maxValue.
  (Scalar minValue, Scalar maxValue) =>
  -- | minimum value
  minValue ->
  -- | maximum value
  maxValue ->
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
hardtanh minValue maxValue tensor = unsafePerformIO $ cast3 ATen.hardtanh_tss tensor minValue maxValue

-- | Applies the hardswish function element-wise.
hardswish ::
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
hardswish = unsafePerformIO . cast1 ATen.hardswish_t

-- | Applies the exponential linear unit function element-wise, with alpha input,
-- \[
-- \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1)).
-- \]
elu ::
  forall alpha.
  Scalar alpha =>
  -- | alpha value for ELU formulation
  alpha ->
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
elu alpha tensor = unsafePerformIO $ cast2 ATen.elu_ts tensor alpha

-- | Applies the scaled exponential linear unit function element-wise, that is,
-- \[
-- \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)),
-- \]
-- with \(\alpha = 1.6732632423543772848170429916717\)
-- and \(\text{scale}=1.0507009873554804934193349852946\).
selu ::
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
selu = unsafePerformIO . cast1 ATen.selu_t

-- | Applies the continuously differentiable exponential linear unit function element-wise, that is,
-- \[
-- \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1)).
-- \]
celu ::
  forall alpha.
  (Scalar alpha) =>
  -- | alpha
  alpha ->
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
celu alpha tensor = unsafePerformIO $ cast2 ATen.celu_ts tensor alpha

-- | Applies the element-wise function:
-- \[
-- \text{LeakyReLU}(x) = \max(0,x) + \text{negativeSlope} * \min(0,x),
-- \]
-- the the angle of the negative slope can be controlled.
-- A typical value for it is 0.01.
leakyRelu ::
  forall negativeSlope.
  (Scalar negativeSlope) =>
  -- | negative slope
  negativeSlope ->
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
leakyRelu negativeSlope tensor = unsafePerformIO $ cast2 ATen.leaky_relu_ts tensor negativeSlope

-- | Applies the parameterized rectified linear unit function element-wise, that is,
-- \[
-- \text{PReLU}(x) = max(0, x) + \text{weight} * min(0, x).
-- \]
-- The weight parameter is typically learnable.
prelu ::
  -- | weight (typically learnable)
  UntypedParameter ->
  -- | input
  UntypedTensor ->
  -- | output
  UntypedTensor
prelu weight tensor = unsafePerformIO $ cast2 ATen.prelu_tt tensor weight
