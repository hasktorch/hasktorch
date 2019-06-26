{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Parameters
import LinearLayer
import RecurrentLayer
import MLP

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

num_iters = 10000
num_timesteps = 3

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith (\p dp -> p - (lr * dp)) (map toDependent parameters) gradients

runOverTimesteps :: Tensor -> Recurrent -> Int -> Tensor -> Tensor
runOverTimesteps inp layer 0 hidden = hidden
runOverTimesteps inp layer n hidden = 
    runOverTimesteps inp layer (n-1) $ recurrent layer inp' hidden
    where
        inp' = reshape (select inp 0 (n-1)) [1, 2]

main :: IO ()
main = do
    rnnLayer <- sample $ RecurrentSpec { in_features = 2, hidden_features = 2, nonlinearitySpec = Torch.Functions.tanh }

    let foldLoop x count block = foldM block x [1..count]

    -- randomly initializing training values
    inp <- randn' [num_timesteps, 2]
    init_hidden <- randn' [1, 2]
    expected_output <- randn' [1, 2]

    -- training loop
    foldLoop rnnLayer num_iters $ \model i -> do

        let output = runOverTimesteps inp model num_timesteps init_hidden
        -- print output

        let loss = mse_loss output expected_output

        let flat_parameters = flattenParameters model

        let gradients = grad loss flat_parameters

        if i `mod` 100 == 0
           then do putStrLn $ show loss
          else return ()

        new_flat_parameters <- mapM makeIndependent $ sgd 5e-2 flat_parameters gradients

        return $ replaceParameters model $ new_flat_parameters

    return ()
