{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Tensor.Other where

import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast2, cast3)
import qualified Torch.Internal.Managed.Native as ATen

-- | triu
triu ::
  forall gradient layout device dataType shape.
  -- | diagonal
  Int ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  Tensor gradient layout device dataType shape
triu diagonal input = unsafePerformIO $ cast2 ATen.triu_tl input diagonal

-- | tril
tril ::
  forall gradient layout device dataType shape.
  -- | diagonal
  Int ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  Tensor gradient layout device dataType shape
tril diagonal input = unsafePerformIO $ cast2 ATen.tril_tl input diagonal

-- | masked fill
maskedFill ::
  forall gradient layout device dataType shape value gradient' layout' device' dataType' shape'.
  (Scalar value, Catch (gradient <+> 'Gradient 'WithoutGradient), Catch (dataType <+> 'DataType 'Bool)) =>
  -- | mask
  Tensor gradient layout device dataType shape ->
  -- | value
  value ->
  -- | input
  Tensor gradient' layout' device' dataType' shape' ->
  -- | output
  Tensor
    gradient'
    (layout <+> layout' <+> 'Layout 'Dense)
    (device <+> device')
    dataType'
    (BroadcastShapesF shape shape')
maskedFill mask value input = unsafePerformIO $ cast3 ATen.masked_fill_tts input mask value
