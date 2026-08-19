{-# LANGUAGE RankNTypes #-}

module Torch.GraduallyTyped.NN.Functional.Activation where

import Control.Monad.Catch (MonadThrow)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (addScalar, mulScalar, powScalar, tanh)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.Internal.Cast (cast1, cast2, cast3)
import qualified Torch.Internal.Managed.Native as ATen
import Torch.Scalar (Scalar)
import Prelude (Float, pure, ($), (*), (+), (.), (/))
import qualified Prelude (pi, sqrt)
import Torch.Internal.GC (unsafeThrowableIO)

-- $setup
-- >>> import Torch.GraduallyTyped.Prelude.List (SList (..))
-- >>> import Torch.GraduallyTyped

-- | Thresholds each element of the input Tensor.
threshold ::
  forall threshold value gradient layout device dataType shape m.
  (Scalar threshold, Scalar value, MonadThrow m) =>
  -- | threshold
  threshold ->
  -- | value
  value ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  m (Tensor gradient layout device dataType shape)
threshold thresholdValue value tensor =
  unsafeThrowableIO $ cast3 ATen.threshold_tss tensor thresholdValue value

-- | Applies the rectified linear unit function element-wise, that is,
-- \[
-- \text{ReLU}(x) = max(0, x).
-- \]
relu ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  Tensor gradient layout device dataType shape
relu = unsafePerformIO . cast1 ATen.relu_t

-- | Applies the gaussian error linear unit function element-wise.
gelu ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  Tensor gradient layout device dataType shape
gelu = unsafePerformIO . cast1 ATen.gelu_t

-- | Applies the gaussian error linear unit function element-wise.
--
-- This is the implementation of the GELU activation function from
-- Google's BERT repo (and coincidentally also from OpenAI's GPT).
-- See also https://arxiv.org/abs/1606.08415.
--
-- >>> t <- sFull (TensorSpec (SGradient SWithGradient) (SLayout SDense) (SDevice SCPU) (SDataType SFloat) (SShape $ SNil)) (0.5 :: Prelude.Float)
-- >>> t' <- geluNew t
-- >>> fromTensor @Float t'
-- 0.345714
geluNew ::
  forall gradient layout device dataType shape m.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  m (Tensor gradient layout device dataType shape)
geluNew x = do
  xHalfed <- x `mulScalar` (0.5 :: Float)
  xCubed <- x `powScalar` (3.0 :: Float)
  xCubedScaled <- xCubed `mulScalar` (0.044715 :: Float)
  x' <- (x + xCubedScaled) `mulScalar` Prelude.sqrt ((2 :: Float) / Prelude.pi)
  x'' <- tanh x' `addScalar` (1 :: Float)
  pure $ xHalfed * x''

-- | Applies the HardTanh function element-wise.
hardtanh ::
  forall minValue maxValue gradient layout device dataType shape m.
  (Scalar minValue, Scalar maxValue, MonadThrow m) =>
  -- | minimum value
  minValue ->
  -- | maximum value
  maxValue ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  m (Tensor gradient layout device dataType shape)
hardtanh minValue maxValue tensor = unsafeThrowableIO $ cast3 ATen.hardtanh_tss tensor minValue maxValue

-- | Applies the hardswish function element-wise.
hardswish ::
  forall gradient layout device dataType shape.
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  Tensor gradient layout device dataType shape
hardswish = unsafePerformIO . cast1 ATen.hardswish_t

-- | Applies the exponential linear unit function element-wise, with alpha input,
-- \[
-- \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1)).
-- \]
elu ::
  forall alpha gradient layout device dataType shape m.
  (Scalar alpha, MonadThrow m) =>
  -- | alpha value for ELU formulation
  alpha ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  m (Tensor gradient layout device dataType shape)
elu alpha tensor = unsafeThrowableIO $ cast2 ATen.elu_ts tensor alpha

-- | Applies the scaled exponential linear unit function element-wise, that is,
-- \[
-- \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)),
-- \]
-- with \(\alpha = 1.6732632423543772848170429916717\)
-- and \(\text{scale}=1.0507009873554804934193349852946\).
selu ::
  forall gradient layout device dataType shape m.
  MonadThrow m =>
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  m (Tensor gradient layout device dataType shape)
selu = unsafeThrowableIO . cast1 ATen.selu_t

-- | Applies the continuously differentiable exponential linear unit function element-wise, that is,
-- \[
-- \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1)).
-- \]
celu ::
  forall alpha gradient layout device dataType shape m.
  (Scalar alpha, MonadThrow m) =>
  -- | alpha
  alpha ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  m (Tensor gradient layout device dataType shape)
celu alpha tensor = unsafeThrowableIO $ cast2 ATen.celu_ts tensor alpha

-- | Applies the element-wise function:
-- \[
-- \text{LeakyReLU}(x) = \max(0,x) + \text{negativeSlope} * \min(0,x),
-- \]
-- the the angle of the negative slope can be controlled.
-- A typical value for it is 0.01.
leakyRelu ::
  forall negativeSlope gradient layout device dataType shape m.
  (Scalar negativeSlope, MonadThrow m) =>
  -- | negative slope
  negativeSlope ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  m (Tensor gradient layout device dataType shape)
leakyRelu negativeSlope tensor = unsafeThrowableIO $ cast2 ATen.leaky_relu_ts tensor negativeSlope

-- | Applies the parameterized rectified linear unit function element-wise, that is,
-- \[
-- \text{PReLU}(x) = max(0, x) + \text{weight} * min(0, x).
-- \]
-- The weight parameter is typically learnable.
prelu ::
  forall gradient' gradient layout device dataType shape m.
  MonadThrow m =>
  -- | weight (typically learnable)
  Tensor gradient' layout device dataType shape ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  m (Tensor gradient layout device dataType shape)
prelu weight tensor = unsafeThrowableIO $ cast2 ATen.prelu_tt tensor weight
