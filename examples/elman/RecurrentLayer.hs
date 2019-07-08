{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module RecurrentLayer where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

------------- General Recurrent Structures ------------------
class RecurrentCell a where
  -- cell ff function written by hand,
  -- to demonstrate how the cell works
  nextState :: a -> Tensor -> Tensor -> Tensor
  -- TODO: there should also be a `forward` function here
  -- that uses the rnn forward functions from ATen
  -- but I'll implement that when I can make sense
  -- of the ATen function arguments


data RecurrentGate = RecurrentGate {
  inputWt :: Parameter,
  hiddenWt :: Parameter,
  bias :: Parameter
  nonLinearity :: Tensor -> Tensor
}

gate :: RecurrentGate -> Tensor -> Tensor -> Tensor
gate RecurrentGate{..} input hidden = 
  nonLinearity $ (mul input inputWt) + (mul hidden hiddenWt) + bias 
  where
    mul features wts = matmul (toDependent weight) (transpose2D features)
------------------------------------------------------------


-- Specifying the shape of the recurrent layer
data RecurrentSpec = RecurrentSpec { in_features :: Int, hidden_features :: Int, nonlinearitySpec :: Tensor -> Tensor }

----------------- Elman Cell -------------------
type ElmanCell = RecurrentGate                            }

instance RecurrentCell ElmanCell where
  nextState cell input hidden = gate cell input hidden
------------------------------------------------

----------------- LSTM Cell ---------------------
data LSTMCell = LSTMCell {
  input_gate   :: RecurrentGate,
  forget_gate  :: RecurrentGate,
  output_gate  :: RecurrentGate,
  hidden_gate  :: RecurrentGate,
  cell_state   :: Parameter 
}

newCellState :: LSTMCell -> Tensor -> Tensor -> Tensor
newCellState LSTMCell{..} input hidden =
  (fg * cell_state) + (ig * c')
  where
    ig = gate input_gate input hidden
    fg = gate forget_gate input hidden
    c' = gate hidden_gate input hidden

instance RecurrentCell LSTMCell where
  nextState cell input hidden =
    og * (Torch.Functions.tanh cNew)  
    where
      og = gate (output_gate cell) input hidden
      cNew = newCellState cell input hidden
------------------------------------------------

------------------ GRU Cell ----------------------
data GRUCell = GRUCell {
  reset_gate :: RecurrentGate,
  update_gate :: RecurrentGate,
  hidden_gate :: RecurrentGate
}

instance RecurrentCell GRUCell where
  nextState GRUCell{..} input hidden =
    (ug * hidden) + ((1 - ug) * h')
    where 
      rg = gate reset_gate input hidden
      ug = gate update_gate input hidden
      h' = gate hidden_gate input (rg * hidden)
--------------------------------------------------
