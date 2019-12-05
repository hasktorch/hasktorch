{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Autograd where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified LibTorch.Torch.Managed.Autograd
import qualified LibTorch.ATen.Managed.Type.Tensor as ATen
import qualified LibTorch.ATen.Type as ATen
import LibTorch.ATen.Class
import LibTorch.ATen.Cast

import Torch.Tensor

newtype IndependentTensor = IndependentTensor { toDependent :: Tensor }
    deriving (Show)

grad :: Tensor -> [IndependentTensor] -> [Tensor]
grad y inputs = unsafePerformIO $ (cast2 LibTorch.Torch.Managed.Autograd.grad) y (map toDependent inputs)

requiresGrad :: Tensor -> Bool
requiresGrad t = unsafePerformIO $ (cast1 ATen.tensor_requires_grad) t

makeIndependent :: Tensor -> IO IndependentTensor
makeIndependent t = (cast1 LibTorch.Torch.Managed.Autograd.makeIndependent) t >>= return . IndependentTensor
