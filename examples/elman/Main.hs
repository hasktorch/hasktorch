{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN
import GHC.Generics

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)


{- This is the core of the RNN example- an Elman Cell type -}

-- Specifying the shape of the recurrent layer
data RecurrentSpec = RecurrentSpec { in_features :: Int, hidden_features :: Int }

-- Recurrent layer type holding the weights for the layer
data Recurrent = Recurrent { weight_ih :: Parameter, 
                             weight_hh :: Parameter,
                             bias :: Parameter
                           } deriving (Generic)


-- Typeclass that shows that the layer weights can be randomly initialized
instance Randomizable RecurrentSpec Recurrent where
  sample RecurrentSpec{..} = do
      w_ih <- makeIndependent =<< randn' [in_features, hidden_features]
      w_hh <- makeIndependent =<< randn' [hidden_features, hidden_features]
      b <- makeIndependent =<< randn' [1, hidden_features]
      return $ Recurrent w_ih w_hh b


-- Typeclass that allows us to manipulate and update the layer weights
instance Parameterized Recurrent where
  flattenParameters Recurrent{..} = [weight_ih, weight_hh, bias]
  replaceOwnParameters _ = do
    weight_ih <- nextParameter
    weight_hh <- nextParameter
    bias   <- nextParameter
    return $ Recurrent{..}

instance Show Recurrent where
  show Recurrent{..} =
    (show $ toDependent weight_ih) ++ "\n" ++
    (show $ toDependent weight_hh) ++ "\n" ++
    (show $ toDependent bias)


-- The layer 'function', i.e: passes an input through the layer
-- and returns output
recurrent :: Recurrent  -- current state of layer
          -> Tensor     -- input tensor
          -> Tensor     -- previous hidden state
          -> Tensor     -- output tensor
recurrent Recurrent{..} hidden input =
  h' (linear weight_ih input)
     (linear weight_hh hidden)
     bias

  where

    linear weight features = 
      matmul
        (toDependent weight)
        (transpose2D features)

    -- combined result of input and hidden gate
    h' ig hg bias = transpose2D 
                    (Torch.Functions.tanh 
                        (ig + hg + 
                            (transpose2D $ toDependent bias)))


-- Running the same layer over multiple timesteps
-- where no. of timesteps <= length of input sequence
runOverTimesteps :: Tensor -> Recurrent -> Tensor -> Tensor
runOverTimesteps inp layer hidden = 
  foldl (recurrent layer) hidden inpList
  where
    -- converting matrix into a list of tensors
    -- this hack stays until I can write a Foldable instance
    -- for a tensor
    inpList = [reshape (inp @@ x) [1, 2] | x <- [0.. ((size inp 0) - 1)]]

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

num_iters = 10000
num_timesteps = 3


main :: IO ()
main = do
    -- randomly initialize the elman cell
    rnnLayer <- sample $ RecurrentSpec { in_features = 2, hidden_features = 2 }

    let foldLoop x count block = foldM block x [1..count]

    -- randomly initializing training values
    inp <- randn' [num_timesteps, 2]
    init_hidden <- randn' [1, 2]
    expected_output <- randn' [1, 2]

    -- training loop
    foldLoop rnnLayer num_iters $ \model i -> do

        -- calculate output when RNN is run over timesteps
        let output = runOverTimesteps inp model init_hidden

        let loss = mse_loss output expected_output

        -- "flatten" parameters into a single list to make it
        -- easier for libtorch grad to work with
        let flat_parameters = flattenParameters model

        -- gradients using libtorch grad functions
        let gradients = grad loss flat_parameters

        -- print loss every 100 iterations
        -- if the RNN is working, loss should reduce
        if i `mod` 100 == 0
            then do putStrLn $ show loss
          else return ()

        -- new parameters returned by the SGD update functions
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-2 flat_parameters gradients

        -- return the new model state "to" the next iteration of foldLoop
        return $ replaceParameters model $ new_flat_parameters

    return ()
