{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Functional.Activation where

import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Type.Equality (type (==))
import GHC.TypeLits (Nat, Symbol)
import GHC.TypeLits.KnownNat (KnownBool, boolVal)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType)
import Torch.GraduallyTyped.Tensor(Tensor)
import Torch.Internal.Cast (cast1, cast2, cast3)
import qualified Torch.Internal.Managed.Native as ATen
import Torch.Scalar (Scalar)

-- | Thresholds each element of the input Tensor.
threshold ::
  forall threshold value requiresGradient layout device dataType shape.
  (Scalar threshold, Scalar value) =>
  -- | threshold
  threshold ->
  -- | value
  value ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
threshold threshold value tensor =
  unsafePerformIO $ cast3 ATen.threshold_tss tensor threshold value

-- | Applies the rectified linear unit function element-wise, that is,
-- \[
-- \text{ReLU}(x) = max(0, x).
-- \]
relu ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
relu = unsafePerformIO . cast1 ATen.relu_t

-- | Applies the HardTanh function element-wise.
hardtanh ::
  forall minValue maxValue requiresGradient layout device dataType shape.
  (Scalar minValue, Scalar maxValue) =>
  -- | minimum value
  minValue ->
  -- | maximum value
  maxValue ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
hardtanh minValue maxValue tensor = unsafePerformIO $ cast3 ATen.hardtanh_tss tensor minValue maxValue

-- | Applies the hardswish function element-wise.
hardswish ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
hardswish = unsafePerformIO . cast1 ATen.hardswish_t

-- | Applies the exponential linear unit function element-wise, with alpha input,
-- \[
-- \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1)).
-- \]
elu ::
  forall alpha requiresGradient layout device dataType shape.
  Scalar alpha =>
  -- | alpha value for ELU formulation
  alpha ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
elu alpha tensor = unsafePerformIO $ cast2 ATen.elu_ts tensor alpha

-- | Applies the scaled exponential linear unit function element-wise, that is,
-- \[
-- \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)),
-- \]
-- with \(\alpha = 1.6732632423543772848170429916717\)
-- and \(\text{scale}=1.0507009873554804934193349852946\).
selu ::
  forall requiresGradient layout device dataType shape.
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
selu = unsafePerformIO . cast1 ATen.selu_t

-- | Applies the continuously differentiable exponential linear unit function element-wise, that is,
-- \[
-- \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1)).
-- \]
celu ::
  forall alpha requiresGradient layout device dataType shape.
  (Scalar alpha) =>
  -- | alpha
  alpha ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
celu alpha tensor = unsafePerformIO $ cast2 ATen.celu_ts tensor alpha

-- | Applies the element-wise function:
-- \[
-- \text{LeakyReLU}(x) = \max(0,x) + \text{negativeSlope} * \min(0,x),
-- \]
-- the the angle of the negative slope can be controlled.
-- A typical value for it is 0.01.
leakyRelu ::
  forall negativeSlope requiresGradient layout device dataType shape.
  (Scalar negativeSlope) =>
  -- | negative slope
  negativeSlope ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
leakyRelu negativeSlope tensor = unsafePerformIO $ cast2 ATen.leaky_relu_ts tensor negativeSlope

-- | Applies the parameterized rectified linear unit function element-wise, that is,
-- \[
-- \text{PReLU}(x) = max(0, x) + \text{weight} * min(0, x).
-- \]
-- The weight parameter is typically learnable.
prelu ::
  forall requiresGradient' requiresGradient layout device dataType shape.
  -- | weight (typically learnable)
  Tensor requiresGradient' layout device dataType shape ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
prelu weight tensor = unsafePerformIO $ cast2 ATen.prelu_tt tensor weight
