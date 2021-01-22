{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Tensor.Other where

import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.Internal.Cast (cast2, cast3)
import qualified Torch.Internal.Managed.Native as ATen
import Torch.GraduallyTyped.Layout (Layout(..), LayoutType(..))

-- | triu
triu ::
  forall requiresGradient layout device dataType shape.
  -- | diagonal
  Int ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
triu diagonal input = unsafePerformIO $ cast2 ATen.triu_tl input diagonal

-- | tril
tril ::
  forall requiresGradient layout device dataType shape.
  -- | diagonal
  Int ->
  -- | input
  Tensor requiresGradient layout device dataType shape ->
  -- | output
  Tensor requiresGradient layout device dataType shape
tril diagonal input = unsafePerformIO $ cast2 ATen.tril_tl input diagonal

-- | masked fill
maskedFill ::
  forall device shape value requiresGradient' device' dataType' shape'.
  (Scalar value) =>
  -- | mask
  Tensor 'WithoutGradient ('Layout 'Dense) device ( 'DataType 'Bool) shape ->
  -- | value
  value ->
  -- | input
  Tensor requiresGradient' ('Layout 'Dense) device' dataType' shape' ->
  -- | output
  Tensor requiresGradient' ('Layout 'Dense) (device <+> device') dataType' (BroadcastShapesF shape shape')
maskedFill mask value input = unsafePerformIO $ cast3 ATen.masked_fill_tts input mask value
