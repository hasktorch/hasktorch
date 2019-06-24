{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN

import Control.Monad (foldM)
import Data.List (foldl', scanl', intersperse)

{- Types -}

data LinearSpec = LinearSpec { in_features :: Int, out_features :: Int }
  deriving (Show, Eq)

data Linear = Linear { weight :: Parameter, bias :: Parameter } deriving Show

{- Instances -}

instance Randomizable LinearSpec Linear where
  sample LinearSpec{..} = do
      w <- makeIndependent =<< randn' [in_features, out_features]
      b <- makeIndependent =<< randn' [out_features]
      return $ Linear w b

instance Parameterized Linear where
  flattenParameters Linear{..} = [weight, bias]
  replaceOwnParameters _ = do
    weight <- nextParameter
    bias <- nextParameter
    return $ Linear{..}

batch_size = 32
num_iters = 10000

model :: Linear -> Tensor -> Tensor
model params t = (linear params t)

linear :: Linear -> Tensor -> Tensor
linear Linear{..} input = (matmul input (toDependent weight)) + (toDependent bias)

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith (\p dp -> p - (lr * dp)) (map toDependent parameters) gradients

main :: IO ()
main = do
    init <- sample $ LinearSpec { in_features = 3, out_features = 1 } 
    trained <- foldLoop init num_iters $ \state i -> do
        input <- rand' [batch_size, 2] >>= return . (toDType Float) . (gt 0.5)
        let expected_output = groundTruth input
        let output = squeezeAll $ model state input
        let loss = mse_loss output expected_output
        let flat_parameters = flattenParameters state
        let gradients = grad loss flat_parameters
        if i `mod` 100 == 0
          then do putStrLn $ show loss
          else return ()
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-4 flat_parameters gradients
        return $ replaceParameters state $ new_flat_parameters
    pure ()
  where
    foldLoop x count block = foldM block x [1..count]
    groundTruth :: Tensor -> Tensor
    groundTruth t = undefined
