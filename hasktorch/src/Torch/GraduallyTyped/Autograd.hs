{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.GraduallyTyped.Autograd where

import Data.Kind (Type)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Tensor
import Torch.Internal.Cast (cast2)
import qualified Torch.Internal.Managed.Autograd as ATen

class HasGrad parameters where
  type Gradients parameters :: Type
  type Loss parameters :: Type

  -- | calculate gradients of a zero-dimensional tensor with respect to a list of independent tensor parameters
  grad :: Loss parameters -> parameters -> Gradients parameters

instance HasGrad (Tensor 'Independent layout device dataType shape) where
  type
    Gradients (Tensor 'Independent layout device dataType shape) =
      Tensor 'Dependent layout device dataType shape
  type
    Loss (Tensor 'Independent layout device dataType shape) =
      Tensor 'Dependent layout device dataType shape

  grad loss parameter = head . unsafePerformIO $ cast2 ATen.grad loss [parameter]
