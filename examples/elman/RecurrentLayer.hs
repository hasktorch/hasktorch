{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module RecurrentLayer where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import ATen.Cast
import qualified ATen.Managed.Native as ATen

import System.IO.Unsafe
import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Parameters
import Utils

{- This is the core of the RNN example- an Elman Cell type -}


-- Specifying the shape of the recurrent layer
data RecurrentSpec = RecurrentSpec { in_features :: Int, hidden_features :: Int, nonlinearitySpec :: Tensor -> Tensor }

-- Recurrent layer type holding the weights for the layer
data Recurrent = Recurrent { weight_ih :: Parameter, bias_ih :: Parameter,
                             weight_hh :: Parameter, bias_hh :: Parameter
                        }


-- Typeclass that shows that the layer weights can be randomly initialized
instance Randomizable RecurrentSpec Recurrent where
  sample RecurrentSpec{..} = do
      w_ih <- makeIndependent =<< randn' [in_features, hidden_features]
      b_ih <- makeIndependent =<< randn' [1, in_features]
      w_hh <- makeIndependent =<< randn' [hidden_features, hidden_features]
      b_hh <- makeIndependent =<< randn' [1, hidden_features]
      return $ Recurrent w_ih b_ih w_hh b_hh


-- Typeclass that allows us to manipulate and update the layer weights
instance Parametrized Recurrent where
  flattenParameters Recurrent{..} = [weight_ih, bias_ih, weight_hh, bias_hh]
  replaceOwnParameters _ = do
    weight_ih <- nextParameter
    bias_ih   <- nextParameter
    weight_hh <- nextParameter
    bias_hh   <- nextParameter
    return $ Recurrent{..}

instance Show Recurrent where
  show Recurrent{..} =
    (show $ toDependent weight_ih) ++ "\n" ++
    (show $ toDependent bias_ih) ++ "\n" ++
    (show $ toDependent weight_hh) ++ "\n" ++
    (show $ toDependent bias_hh)


-- The layer 'function', i.e: passes an input through the layer
-- and returns output
recurrent :: Recurrent  -- current state of layer
          -> Tensor     -- input tensor
          -> Tensor     -- previous hidden state
          -> Tensor     -- output tensor
recurrent Recurrent{..} input hidden =
  h' (inputGate weight_ih bias_ih input)
     (hiddenGate weight_hh bias_hh hidden)

  where
    -- Input gate
    inputGate weight bias input =
      (matmul
        (toDependent weight)
        (transpose2D input)) +
      (transpose2D $ toDependent bias)
    -- hidden gate
    hiddenGate weight bias hidden =
      (matmul
        (toDependent weight)
        (transpose2D hidden)) +
      (transpose2D $ toDependent bias)
    -- combined result of input and hidden gate
    h' ig hg = transpose2D $ Torch.Functions.tanh (ig + hg)


-- Running the same layer over multiple timesteps
-- where no. of timesteps <= length of input sequence
runOverTimesteps :: Tensor -> Recurrent -> Int -> Tensor -> Tensor
runOverTimesteps inp layer 0 hidden = hidden
runOverTimesteps inp layer n hidden =
    runOverTimesteps inp layer (n-1) $ recurrent layer inp' hidden
    where
        inp' = reshape (select inp 0 (n-1)) [1, 2]
