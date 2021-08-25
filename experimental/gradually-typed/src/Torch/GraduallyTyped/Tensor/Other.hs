{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Tensor.Other where

import Control.Monad.Catch (MonadThrow)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import qualified Torch.Internal.Cast as ATen (cast2, cast3)
import Torch.Internal.GC (unsafeThrowableIO)
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
triu diagonal input = unsafePerformIO $ ATen.cast2 ATen.triu_tl input diagonal

-- | tril
tril ::
  forall gradient layout device dataType shape.
  -- | diagonal
  Int ->
  -- | input
  Tensor gradient layout device dataType shape ->
  -- | output
  Tensor gradient layout device dataType shape
tril diagonal input = unsafePerformIO $ ATen.cast2 ATen.tril_tl input diagonal

-- | masked fill
maskedFill ::
  forall gradient layout device dataType shape value gradient' layout' device' dataType' shape' shape'' m.
  ( Scalar value,
    MonadThrow m,
    Catch (gradient <+> 'Gradient 'WithoutGradient),
    Catch (dataType <+> 'DataType 'Bool),
    shape'' ~ BroadcastShapesF shape shape',
    Catch shape''
  ) =>
  -- | mask
  Tensor gradient layout device dataType shape ->
  -- | value
  value ->
  -- | input
  Tensor gradient' layout' device' dataType' shape' ->
  -- | output
  m
    ( Tensor
        gradient'
        (layout <+> layout' <+> 'Layout 'Dense)
        (device <+> device')
        dataType'
        shape''
    )
maskedFill mask value input = unsafeThrowableIO $ ATen.cast3 ATen.masked_fill_tts input mask value
