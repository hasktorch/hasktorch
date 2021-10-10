{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.NN.Functional.Loss where

import Control.Monad.Catch (MonadThrow)
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.Shape.Type (Shape (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import qualified Torch.Internal.Cast as ATen
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Native as ATen

-- | Compute the mean squared error between two tensors.
mseLoss ::
  forall m gradient layout device dataType shape gradient' layout' device' dataType' shape'.
  (MonadThrow m, Catch (shape <+> shape')) =>
  -- | prediction tensor
  Tensor gradient layout device dataType shape ->
  -- | target tensor
  Tensor gradient' layout' device' dataType' shape' ->
  -- | output tensor
  m
    ( Tensor
        (gradient <|> gradient')
        (layout <+> layout')
        (device <+> device')
        (dataType <+> dataType')
        ('Shape '[])
    )
prediction `mseLoss` target =
  unsafeThrowableIO $
    ATen.cast3
      ATen.mse_loss_ttl
      prediction
      target
      (1 :: Int) -- reduce mean
