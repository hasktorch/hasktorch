{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Utils where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import ATen.Cast
import qualified ATen.Managed.Native as ATen

import System.IO.Unsafe
import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Parameters

-- more of an update function- updates the parameters using SGD
-- update rule when given parameters, learning rate and gradients
sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith (\p dp -> p - (lr * dp)) (map toDependent parameters) gradients

-- Tranpose functions- "haskellized" from the FFI bindings
-- TODO: these need to go in ffi-experimental/hasktorch
transpose :: Tensor -> Int -> Int -> Tensor
transpose t a b = unsafePerformIO $ (cast3 ATen.transpose_tll) t a b

transpose2D :: Tensor -> Tensor
transpose2D t = transpose t 0 1

transpose1D :: Tensor -> Tensor
transpose1D t = reshape t  [(head $ shape t), 1]

