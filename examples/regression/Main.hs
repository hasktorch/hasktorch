{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (when)
import Torch.Autograd (grad, makeIndependent, toDependent)
import Torch.Device (DeviceType (CPU), Device (Device))
import Torch.DType (DType (Float))
import Torch.Functional (squeezeAll, matmul, mse_loss)
import Torch.NN
  ( Linear (Linear, bias, weight),
    LinearSpec (LinearSpec, in_features, out_features),
    flattenParameters,
    linear,
    replaceParameters,
    sample
  )
import Torch.Optim (foldLoop, GD(..), runStep, sgd)
import Torch.Random (mkGenerator, rand', randn')
import Torch.Tensor (Tensor, asTensor)
import Torch.TensorFactories (full', ones')

model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias = full' [1] (3.14 :: Float)
    
printParams :: Linear -> IO ()
printParams trained = do
    putStrLn $ "Parameters:\n" ++ (show $ toDependent $ weight trained)
    putStrLn $ "Bias:\n" ++ (show $ toDependent $ bias trained) 

main :: IO ()
main = do
    init <- sample $ LinearSpec { in_features = numFeatures, out_features = 1 } 
    randGen <- defaultRNG
    printParams init
    (trained, _) <- foldLoop (init, randGen) numIters $ \(state, randGen) i -> do
        let (input, randGen') = randn' [batchSize, numFeatures] randGen
            (y, y') = (groundTruth input, model state input)
            loss = mse_loss y' y
        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
        (newParam, _) <- runStep state optimizer loss 5e-3 
        pure (replaceParameters state newParam, randGen')
    printParams trained
    pure ()
  where
    optimizer = GD
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 4
    numIters = 2000
    numFeatures = 3
