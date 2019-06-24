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

num_iters = 100
num_timesteps = 3

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith (\p dp -> p - (lr * dp)) (map toDependent parameters) gradients

main :: IO ()
main = do
    rnnLayer <- sample $ RecurrentSpec { in_features = 2, hidden_features = 2, nonlinearitySpec = Torch.Functions.tanh }

    let foldLoop x count block = foldM block x [1..count]

    -- randomly initializing training values
    inp <- randn' [num_timesteps, 2]
    init_hidden <- randn' [2]
    expected_output <- randn' [1, 2]

    let init_hidden' = reshape init_hidden [1, 2]
    let myRNN = RunRecurrent rnnLayer [init_hidden']

    -- training
    trained <- foldLoop myRNN num_iters $ \model i -> do

        -- running the thing over n timesteps
        model_after_timesteps <-
            foldLoop myRNN num_timesteps $ \model i -> do

                let inp' = reshape (select inp 0 (i-1)) [1, 2]
                let hidden = head $ past model

                let out' = recurrent (rnn model) inp' hidden

                let model' = RunRecurrent (rnn model) (out' : (past model))

                return model'
        -------------------------------------

        let output = head $ past model_after_timesteps

        let loss = mse_loss output expected_output

        let flat_parameters = flattenParameters $ rnn model

        let gradients = grad loss flat_parameters

        -- if i `mod` 10 == 0
        --   then do putStrLn $ show loss
        --  else return ()

        new_flat_parameters <- mapM makeIndependent $ sgd 5e-4 flat_parameters gradients

        let rnn' = replaceParameters (rnn model) $ new_flat_parameters
        let updated_model = RunRecurrent rnn' [init_hidden']

        return updated_model

    return ()
