{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM, when)
import Torch.Autograd (grad, makeIndependent, toDependent)
import Torch.DType (DType (Float))
import Torch.Functional (squeezeAll, matmul, mse_loss)
import Torch.NN
  ( Linear (Linear, bias, weight),
    LinearSpec (LinearSpec, in_features, out_features),
    flattenParameters,
    linear,
    replaceParameters,
    sample,
    sgd,
  )
import Torch.Tensor (Tensor, asTensor)
import Torch.TensorFactories (ones', rand', randn')

batch_size = 64
num_iters = 2000
num_features = 3

model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
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
        when (i `mod` 100 == 0) do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-3 flat_parameters gradients
        return $ replaceParameters state $ new_flat_parameters
    printParams trained
    pure ()
  where
    foldLoop x count block = foldM block x [1..count]
