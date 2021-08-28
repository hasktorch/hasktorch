{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module GRU where

import Control.Monad.State.Strict
import Data.List (foldl', intersperse, scanl')
import RecurrentLayer
import Torch

-- Specifying the shape of the recurrent layer
data GRUSpec = GRUSpec {in_f :: Int, h_f :: Int}

data GRUCell = GRUCell
  { reset_gate :: [Parameter],
    update_gate :: [Parameter],
    gru_hidden_gate :: [Parameter]
  }

instance RecurrentCell GRUCell where
  nextState GRUCell {..} input hidden =
    (ug * hidden) + ((1 - ug) * h')
    where
      rg =
        gate
          input
          hidden
          sigmoid
          (reset_gate !! 0)
          (reset_gate !! 1)
          (reset_gate !! 2)
      ug =
        gate
          input
          hidden
          sigmoid
          (update_gate !! 0)
          (update_gate !! 1)
          (update_gate !! 2)
      h' =
        gate
          input
          (rg * hidden)
          Torch.tanh
          (gru_hidden_gate !! 0)
          (gru_hidden_gate !! 1)
          (gru_hidden_gate !! 2)

instance Randomizable GRUSpec GRUCell where
  sample GRUSpec {..} = do
    rg_ih <- makeIndependent =<< randnIO' [in_f, h_f]
    rg_hh <- makeIndependent =<< randnIO' [h_f, h_f]
    rg_b <- makeIndependent =<< randnIO' [1, h_f]
    ug_ih <- makeIndependent =<< randnIO' [in_f, h_f]
    ug_hh <- makeIndependent =<< randnIO' [h_f, h_f]
    ug_b <- makeIndependent =<< randnIO' [1, h_f]
    hg_ih <- makeIndependent =<< randnIO' [in_f, h_f]
    hg_hh <- makeIndependent =<< randnIO' [h_f, h_f]
    hg_b <- makeIndependent =<< randnIO' [1, h_f]
    let rg = [rg_ih, rg_hh, rg_b]
    let ug = [ug_ih, ug_hh, ug_b]
    let hg = [hg_ih, hg_hh, hg_b]
    return $ GRUCell rg ug hg

instance Parameterized GRUCell where
  flattenParameters GRUCell {..} =
    reset_gate ++ update_gate ++ gru_hidden_gate
  _replaceParameters _ = do
    rg_ih <- nextParameter
    rg_hh <- nextParameter
    rg_b <- nextParameter
    ug_ih <- nextParameter
    ug_hh <- nextParameter
    ug_b <- nextParameter
    hg_ih <- nextParameter
    hg_hh <- nextParameter
    hg_b <- nextParameter
    let reset_gate = [rg_ih, rg_hh, rg_b]
    let update_gate = [ug_ih, ug_hh, ug_b]
    let gru_hidden_gate = [hg_ih, hg_hh, hg_b]
    return $ GRUCell {..}

instance Show GRUCell where
  show GRUCell {..} =
    (show $ reset_gate) ++ "\n"
      ++ (show $ update_gate)
      ++ "\n"
      ++ (show $ gru_hidden_gate)
