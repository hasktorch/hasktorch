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

num_iters = 3

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith (\p dp -> p - (lr * dp)) (map toDependent parameters) gradients

main :: IO ()
main = do
    rnnLayer <- sample $ RecurrentSpec { in_features = 2, hidden_features = 2, nonlinearitySpec = Torch.Functions.tanh }

    let foldLoop x count block = foldM block x [1..count]

    -- randomly initializing training values
    inp <- randn' [num_iters, 2]
    init_hidden <- randn' [2]
    out <- randn' [1, 2]

    let init_hidden' = reshape init_hidden [1, 2]
    let myRNN = RunRecurrent rnnLayer [init_hidden']

    -- running the thing
    output <- foldLoop myRNN num_iters $ \model i -> do


            print model
            -- putStrLn $ "FF timestep" ++ (show i) 
            -- print $ rnn model
            -- putStrLn "***"

            -- print inp
            -- print hid
            -- print out
            -- print $ select inp 0 (i-1)
            -- putStrLn "***"

            let inp' = reshape (select inp 0 (i-1)) [1, 2]
            let hidden = head $ past model
        
            let out' = recurrent (rnn model) inp' hidden

            -- print out'
            let model' = RunRecurrent (rnn model) (out' : (past model))

            putStrLn "--------------------------------------------"

            return model'

    print output
