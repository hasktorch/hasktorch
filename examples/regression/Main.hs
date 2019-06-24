{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM)

import Torch.Tensor (Tensor)
import Torch.DType (DType (Float))
import Torch.TensorFactories (ones', rand', randn')
import Torch.Functions
import Torch.Autograd
import Torch.NN

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

{- Forward functions -}

linear :: Linear -> Tensor -> Tensor
linear Linear{..} input = (matmul input dWeight) + dBias
  where
    (dWeight, dBias) = (toDependent weight, toDependent bias)
  
{- Optimization -}

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith step depParameters gradients
  where 
    step p dp = p - (lr * dp)
    depParameters = (map toDependent parameters)

{- Main -}

main :: IO ()
main = do
    init <- sample $ LinearSpec { in_features = num_features, out_features = 1 } 
    trained <- foldLoop init num_iters $ \state i -> do
        input <- rand' [batch_size, num_features] >>= return . (toDType Float)
        let expected_output = squeezeAll $ groundTruth input
        let output = squeezeAll $ linear state input
        let loss = mse_loss output expected_output
        let flat_parameters = flattenParameters state
        let gradients = grad loss flat_parameters
        if i `mod` 500 == 0 then
          putStrLn $ "Loss: " ++ show loss
        else
          pure ()
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-3 flat_parameters gradients
        return $ replaceParameters state $ new_flat_parameters
    putStrLn "Parameters:"
    print $ toDependent $ weight trained
    putStrLn "Bias:"
    print $ toDependent $ bias trained
    pure ()
  where
    batch_size = 64
    num_iters = 10000
    num_features = 3
    foldLoop x count block = foldM block x [1..count]
    groundTruth :: Tensor -> Tensor
    groundTruth = linear Linear { weight = IndependentTensor $ 5.0 * ones' [num_features, 1],
                                    bias = IndependentTensor $ 2.5 * ones' [] }
