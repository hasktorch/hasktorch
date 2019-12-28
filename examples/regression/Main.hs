{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM, when)

import Torch.Autograd
import Torch.Device ( Device(..), DeviceType(..) )
import Torch.DType (DType (Float))
import Torch.Functional
import Torch.NN
import Torch.Tensor
import Torch.TensorFactories (full')
import Torch.Random (mkGenerator, randn')

batch_size = 64
num_iters = 2000
num_features = 3

model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias = full' [1] (3.14 :: Float)
    
printParams :: Linear -> IO ()
printParams trained = do
    putStrLn "Parameters:"
    print $ toDependent $ weight trained
    putStrLn "Bias:"
    print $ toDependent $ bias trained

main :: IO ()
main = do
    init <- sample $ LinearSpec { in_features = num_features, out_features = 1 } 
    randGen <- mkGenerator (Device CPU 0) 31415
    (trained, _) <- foldLoop (init, randGen) num_iters $ \(state, randGen) i -> do
        let (input, randGen') = randn' [batch_size, num_features] randGen
            expected_output = groundTruth input
            output = model state input
            loss = mse_loss output expected_output
            flat_parameters = flattenParameters state
            gradients = grad loss flat_parameters
        when (i `mod` 100 == 0) do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-3 flat_parameters gradients
        pure ((replaceParameters state $ new_flat_parameters), randGen')
    printParams trained
    pure ()
  where
    foldLoop x count block = foldM block x [1..count]
