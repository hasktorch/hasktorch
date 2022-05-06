{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Autograd where

import Foreign.ForeignPtr
import GHC.Generics
import System.IO.Unsafe
import Torch.Internal.Cast
import Torch.Internal.Class
import qualified Torch.Internal.Managed.Autograd
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Tensor

-- | Note: to create an `IndependentTensor` use `makeIndependent`;
-- | otherwise, Torch will complain the parameter does not require a gradient.
newtype IndependentTensor = IndependentTensor
  { toDependent :: Tensor
  }
  deriving (Show, Generic)

grad :: Tensor -> [IndependentTensor] -> [Tensor]
grad y inputs = unsafePerformIO $ cast2 Torch.Internal.Managed.Autograd.grad y (map toDependent inputs)

requiresGrad :: Tensor -> Bool
requiresGrad t = unsafePerformIO $ cast1 ATen.tensor_requires_grad t

setRequiresGrad :: Bool -> Tensor -> Tensor
setRequiresGrad flag t = unsafePerformIO $ cast2 ATen.tensor_set_requires_grad_b t flag

makeIndependent :: Tensor -> IO IndependentTensor
makeIndependent tensor = makeIndependentWithRequiresGrad tensor True

makeIndependentWithRequiresGrad :: Tensor -> Bool -> IO IndependentTensor
makeIndependentWithRequiresGrad tensor requires_grad = IndependentTensor <$> cast2 Torch.Internal.Managed.Autograd.makeIndependent tensor requires_grad
