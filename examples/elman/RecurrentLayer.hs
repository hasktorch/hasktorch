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

-- Specifying the shape of the recurrent layer
data RecurrentSpec = RecurrentSpec { in_features :: Int, hidden_features :: Int, nonlinearitySpec :: Tensor -> Tensor }

class RecurrentCell a where
  -- cell ff function written by hand,
  -- to demonstrate how the cell works
  nextState :: a -> Tensor -> Tensor -> Tensor
  -- TODO: there should also be a `forward` function here
  -- that uses the rnn forward functions from ATen
  -- but I'll implement that when I can make sense
  -- of the ATen function arguments


data Gate = Gate {
  inputWt :: Parameter,
  hiddenWt :: Parameter,
  biasWt :: Parameter,
  nonLinearity :: Tensor -> Tensor
}

instance RecurrentCell Gate where
  nextState Gate{..} input hidden = 
    nonLinearity $ (mul input inputWt) + (mul hidden hiddenWt) + (toDependent biasWt) 
    where
      mul features wts = matmul (toDependent wts) (transpose2D features)


-- Typeclass that shows that the layer weights can be randomly initialized
instance Randomizable RecurrentSpec Gate where
  sample RecurrentSpec{..} = do
      w_ih <- makeIndependent =<< randn' [in_features, hidden_features]
      w_hh <- makeIndependent =<< randn' [hidden_features, hidden_features]
      b <- makeIndependent =<< randn' [1, hidden_features]
      return $ Gate w_ih w_hh b nonlinearitySpec


-- Typeclass that allows us to manipulate and update the layer weights
instance Parameterized Gate where
  flattenParameters Gate{..} = [inputWt, hiddenWt, biasWt]
  replaceOwnParameters _ = do
    inputWt <- nextParameter
    hiddenWt <- nextParameter
    biasWt   <- nextParameter
    return $ Gate{..}

instance Show Gate where
  show Gate{..} =
    (show $ toDependent inputWt) ++ "\n" ++
    (show $ toDependent hiddenWt) ++ "\n" ++
    (show $ toDependent biasWt)
------------------------------------------------------------


----------------- Elman Cell -------------------
type ElmanCell = Gate                            
----------------------------------------------


----------------- LSTM Cell ---------------------
data LSTMCell = LSTMCell {
  input_gate   :: Gate,
  forget_gate  :: Gate,
  output_gate  :: Gate,
  hidden_gate  :: Gate,
  cell_state   :: Parameter 
}

newCellState :: LSTMCell -> Tensor -> Tensor -> Tensor
newCellState LSTMCell{..} input hidden =
  (fg * (toDependent cell_state)) + (ig * c')
  where
    ig = nextState input_gate input hidden
    fg = nextState forget_gate input hidden
    c' = nextState hidden_gate input hidden

instance RecurrentCell LSTMCell where
  nextState cell input hidden =
    og * (Torch.Functions.tanh cNew)  
    where
      og = nextState (output_gate cell) input hidden
      cNew = newCellState cell input hidden

-- Typeclass that shows that the layer weights can be randomly initialized
instance Randomizable RecurrentSpec LSTMCell where
  sample RecurrentSpec{..} = do
      w_ih <- makeIndependent =<< randn' [in_features, hidden_features]
      w_hh <- makeIndependent =<< randn' [hidden_features, hidden_features]
      b <- makeIndependent =<< randn' [1, hidden_features]
      return $ Gate w_ih w_hh b nonlinearitySpec


-- Typeclass that allows us to manipulate and update the layer weights
instance Parameterized Gate where
  flattenParameters Gate{..} = [inputWt, hiddenWt, biasWt]
  replaceOwnParameters _ = do
    inputWt <- nextParameter
    hiddenWt <- nextParameter
    biasWt   <- nextParameter
    return $ Gate{..}

instance Show Gate where
  show Gate{..} =
    (show $ toDependent inputWt) ++ "\n" ++
    (show $ toDependent hiddenWt) ++ "\n" ++
    (show $ toDependent biasWt)

------------------------------------------------

{-
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
-}