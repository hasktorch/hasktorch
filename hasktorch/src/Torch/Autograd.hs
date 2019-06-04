{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Autograd where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified Torch.Managed.Autograd
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Type as ATen
import ATen.Class
import ATen.Cast

import Torch.Tensor

-- NB: ATen only defines Castable [ForeignPtr ATen.Tensor] (ForeignPtr ATen.TensorList)
instance Castable [Tensor] (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list

newtype IndependentTensor = IndependentTensor { toDependent :: Tensor }
    deriving (Show)

grad :: Tensor -> [IndependentTensor] -> [Tensor]
grad y inputs = unsafePerformIO $ (cast2 Torch.Managed.Autograd.grad) y (map toDependent inputs)

requiresGrad :: Tensor -> Bool
requiresGrad t = unsafePerformIO $ (cast1 ATen.tensor_requires_grad) t

makeIndependent :: Tensor -> IO IndependentTensor
makeIndependent t = (cast1 Torch.Managed.Autograd.makeIndependent) t >>= return . IndependentTensor
