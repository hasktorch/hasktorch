{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module LSTM where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import RecurrentLayer


data LSTMSpec = LSTMSpec { inf :: Int, hf :: Int}

data LSTMCell = LSTMCell {
  input_gate   :: [Parameter],
  forget_gate  :: [Parameter],
  output_gate  :: [Parameter],
  hidden_gate  :: [Parameter],
  cell_state   :: Parameter
}


newCellState :: LSTMCell -> Tensor -> Tensor -> Tensor
newCellState LSTMCell{..} input hidden =
  (fg * (toDependent cell_state)) + (ig * c')
  where
    ig = gate input hidden Torch.Functions.sigmoid
         (input_gate !! 0)
         (input_gate !! 1)
         (input_gate !! 2)
    fg = gate input hidden Torch.Functions.sigmoid
         (forget_gate !! 0)
         (forget_gate !! 1)
         (forget_gate !! 2)
    c' = gate input hidden Torch.Functions.sigmoid
         (hidden_gate !! 0)
         (hidden_gate !! 1)
         (hidden_gate !! 2)


instance RecurrentCell LSTMCell where
  nextState cell input hidden =
    matmul og (Torch.Functions.tanh cNew)
    where
      og' = output_gate cell
      og = gate input hidden Torch.Functions.sigmoid
           (og' !! 0)
           (og' !! 1)
           (og' !! 2)
      cNew = newCellState cell input hidden



instance Randomizable LSTMSpec LSTMCell where
  sample LSTMSpec{..} = do
      w_ih <- makeIndependent =<< randn' [inf, hf]
      w_hh <- makeIndependent =<< randn' [hf, hf]
      b <- makeIndependent =<< randn' [1, hf]
      let ig = [w_ih, w_hh, b]
      let fg = [w_ih, w_hh, b]
      let og = [w_ih, w_hh, b]
      let hg = [w_ih, w_hh, b]
      c <- makeIndependent =<< randn' [hf, hf]
      return $ LSTMCell ig fg og hg c


-- Typeclass that allows us to manipulate and update the layer weights
instance Parameterized LSTMCell where
  flattenParameters LSTMCell{..} =
    input_gate ++ forget_gate ++ hidden_gate ++
    output_gate ++ [cell_state]
  replaceOwnParameters _ = do
    ig_ih <- nextParameter
    ig_hh <- nextParameter
    ig_b  <- nextParameter
    fg_ih <- nextParameter
    fg_hh <- nextParameter
    fg_b <- nextParameter
    hg_ih <- nextParameter
    hg_hh <- nextParameter
    hg_b <- nextParameter
    og_ih <- nextParameter
    og_hh <- nextParameter
    og_b <- nextParameter
    cell_state <- nextParameter
    let input_gate = [ig_ih, ig_hh, ig_b]
    let forget_gate = [fg_ih, fg_hh, fg_b]
    let hidden_gate = [hg_ih, hg_hh, hg_b]
    let output_gate = [og_ih, og_hh, og_b]
    return $ LSTMCell{..}

instance Show LSTMCell where
  show LSTMCell{..} =
    (show $ input_gate) ++ "\n" ++
    (show $ forget_gate) ++ "\n" ++
    (show $ output_gate) ++ "\n" ++
    (show $ hidden_gate) ++ "\n" ++
    (show $ cell_state)
