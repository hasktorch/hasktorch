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
num_iters = 10000
num_features = 3

model :: Linear -> Tensor -> Tensor
model Linear{..} input = (matmul input depWeight) + depBias
  where
    (depWeight, depBias) = (toDependent weight, toDependent bias)

groundTruth :: Tensor -> IO Tensor
groundTruth t = do 
    weight <- makeIndependent $ 5.0 * ones' [num_features, 1]
    bias <- makeIndependent $ 2.5 * ones' []
    pure $ model Linear { weight = weight, bias = bias} t

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
        input <- randn' [batch_size, num_features] >>= return . (toDType Float)
        expected_output <- squeezeAll <$> groundTruth input
        let output = squeezeAll $ model state input
            loss = mse_loss output expected_output
            flat_parameters = flattenParameters state
            gradients = grad loss flat_parameters
        if i `mod` 500 == 0 then
          putStrLn $ "Loss: " ++ show loss
        else
          pure ()
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-3 flat_parameters gradients
        return $ replaceParameters state $ new_flat_parameters
    printParams trained
    pure ()
  where
    foldLoop x count block = foldM block x [1..count]
