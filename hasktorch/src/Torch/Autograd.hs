{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Autograd where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified Torch.Managed.Autograd
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Internal.Class
import Torch.Internal.Cast

import Torch.Tensor

newtype IndependentTensor = IndependentTensor { toDependent :: Tensor }
    deriving (Show)

grad :: Tensor -> [IndependentTensor] -> [Tensor]
grad y inputs = unsafePerformIO $ (cast2 Torch.Managed.Autograd.grad) y (map toDependent inputs)

requiresGrad :: Tensor -> Bool
requiresGrad t = unsafePerformIO $ (cast1 ATen.tensor_requires_grad) t

makeIndependent :: Tensor -> IO IndependentTensor
makeIndependent t = (cast1 Torch.Managed.Autograd.makeIndependent) t >>= return . IndependentTensor
