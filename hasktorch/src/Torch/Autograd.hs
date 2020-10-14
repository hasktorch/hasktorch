{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Autograd where

import Foreign.ForeignPtr
import System.IO.Unsafe
import Torch.Internal.Cast
import Torch.Internal.Class
import qualified Torch.Internal.Managed.Autograd
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Tensor

-- | Note: to create an `IndependentTensor` use `makeIndependent`;
-- | otherwise, Torch will complain the parameter does not require a gradient.
newtype IndependentTensor
  = IndependentTensor
      { toDependent :: Tensor
      }
  deriving (Show)

grad :: Tensor -> [IndependentTensor] -> [Tensor]
grad y inputs = unsafePerformIO $ cast2 Torch.Internal.Managed.Autograd.grad y (map toDependent inputs)

requiresGrad :: Tensor -> Bool
requiresGrad t = unsafePerformIO $ cast1 ATen.tensor_requires_grad t

makeIndependent :: Tensor -> IO IndependentTensor
makeIndependent t = IndependentTensor <$> cast1 Torch.Internal.Managed.Autograd.makeIndependent t
