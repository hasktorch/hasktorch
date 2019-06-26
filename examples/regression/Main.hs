{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM)

import Torch.Tensor
import Torch.DType (DType (Float))
import Torch.TensorFactories (ones', rand', randn')
import Torch.Functions
import Torch.Autograd
import Torch.NN

batch_size = 64
num_iters = 2000
num_features = 3

model :: Linear -> Tensor -> Tensor
model Linear{..} input = squeezeAll $ matmul input depWeight + depBias
  where
    (depWeight, depBias) = (toDependent weight, toDependent bias)

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = 42.0 * ones' [num_features, 1]
    bias = 3.14 * ones' [1]
    
printParams :: Linear -> IO ()
printParams trained = do
    putStrLn "Parameters:"
    print $ toDependent $ weight trained
    putStrLn "Bias:"
    print $ toDependent $ bias trained

main :: IO ()
main = do
    init <- sample $ LinearSpec { in_features = num_features, out_features = 1 } 
    trained <- foldLoop init num_iters $ \state i -> do
        input <- randn' [batch_size, num_features]
        let expected_output = groundTruth input
            output = model state input
            loss = mse_loss output expected_output
            flat_parameters = flattenParameters state
            gradients = grad loss flat_parameters
        if i `mod` 100 == 0 then
          putStrLn $ "Loss: " ++ show loss
        else
          pure ()
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-3 flat_parameters gradients
        return $ replaceParameters state $ new_flat_parameters
    printParams trained
    pure ()
  where
    foldLoop x count block = foldM block x [1..count]
