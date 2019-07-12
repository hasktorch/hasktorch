{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

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
-- Specifying the shape of the recurrent layer
data LSTMSpec = LSTMSpec { inf :: Int, hf :: Int}

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
instance Randomizable LSTMSpec LSTMCell where
  sample LSTMSpec{..} = do
      w_ih <- makeIndependent =<< randn' [inf, hf]
      w_hh <- makeIndependent =<< randn' [hf, hf]
      b <- makeIndependent =<< randn' [1, hf]
      let ig = Gate w_ih w_hh b Torch.Functions.sigmoid
      let fg = Gate w_ih w_hh b Torch.Functions.sigmoid
      let og = Gate w_ih w_hh b Torch.Functions.sigmoid
      let hg = Gate w_ih w_hh b Torch.Functions.sigmoid
      c <- makeIndependent =<< randn' [hf, hf]
      return $ LSTMCell ig fg og hg c

-- Typeclass that allows us to manipulate and update the layer weights
instance Parameterized LSTMCell where
  flattenParameters LSTMCell{..} = 
    (flattenParameters input_gate) ++ 
    (flattenParameters forget_gate) ++
    (flattenParameters hidden_gate) ++
    (flattenParameters output_gate) ++
    [cell_state]
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
    let input_gate = Gate ig_ih ig_hh ig_b Torch.Functions.sigmoid
    let forget_gate = Gate fg_ih fg_hh fg_b Torch.Functions.sigmoid
    let hidden_gate = Gate hg_ih hg_hh hg_b Torch.Functions.sigmoid
    let output_gate = Gate og_ih og_hh og_b Torch.Functions.sigmoid
    return $ LSTMCell{..}

instance Show LSTMCell where
  show LSTMCell{..} =
    (show $ input_gate) ++ "\n" ++
    (show $ forget_gate) ++ "\n" ++
    (show $ output_gate) ++ "\n" ++
    (show $ hidden_gate) ++ "\n" ++
    (show $ toDependent cell_state)

------------------------------------------------


------------------ GRU Cell ----------------------
-- Specifying the shape of the recurrent layer
data GRUSpec = GRUSpec { in_f :: Int, h_f :: Int}

data GRUCell = GRUCell {
  reset_gate :: Gate,
  update_gate :: Gate,
  gru_hidden_gate :: Gate
}


instance RecurrentCell GRUCell where
  nextState GRUCell{..} input hidden =
    (ug * hidden) + ((1 - ug) * h')
    where 
      rg = nextState reset_gate input hidden
      ug = nextState update_gate input hidden
      h' = nextState gru_hidden_gate input (rg * hidden)


instance Randomizable GRUSpec GRUCell where
  sample GRUSpec{..} = do
      w_ih <- makeIndependent =<< randn' [in_f, h_f]
      w_hh <- makeIndependent =<< randn' [h_f, h_f]
      b <- makeIndependent =<< randn' [1, h_f]
      let rg = Gate w_ih w_hh b Torch.Functions.sigmoid
      let ug = Gate w_ih w_hh b Torch.Functions.sigmoid
      let hg = Gate w_ih w_hh b Torch.Functions.sigmoid
      return $ GRUCell rg ug hg

instance Parameterized GRUCell where
  flattenParameters GRUCell{..} = 
    (flattenParameters reset_gate) ++ 
    (flattenParameters update_gate) ++
    (flattenParameters gru_hidden_gate)
  replaceOwnParameters _ = do
    rg_ih <- nextParameter
    rg_hh <- nextParameter
    rg_b  <- nextParameter
    ug_ih <- nextParameter
    ug_hh <- nextParameter
    ug_b <- nextParameter
    hg_ih <- nextParameter
    hg_hh <- nextParameter
    hg_b <- nextParameter
    let reset_gate = Gate rg_ih rg_hh rg_b Torch.Functions.sigmoid
    let update_gate = Gate ug_ih ug_hh ug_b Torch.Functions.sigmoid
    let gru_hidden_gate = Gate hg_ih hg_hh hg_b Torch.Functions.sigmoid
    return $ GRUCell{..}

instance Show GRUCell where
  show GRUCell{..} =
    (show $ reset_gate) ++ "\n" ++
    (show $ update_gate) ++ "\n" ++
    (show $ gru_hidden_gate)

--------------------------------------------------
