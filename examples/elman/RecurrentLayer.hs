{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module RecurrentLayer where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Parameters


data RecurrentSpec = RecurrentSpec { in_features :: Int, hidden_features :: Int, nonlinearitySpec :: Tensor -> Tensor }


data Recurrent = Recurrent { weight_ih :: Parameter, bias_ih :: Parameter,
                             weight_hh :: Parameter, bias_hh :: Parameter
                        }


instance Randomizable RecurrentSpec Recurrent where
  sample RecurrentSpec{..} = do
      w_ih <- makeIndependent =<< randn' [in_features, hidden_features]
      b_ih <- makeIndependent =<< randn' [in_features]
      w_hh <- makeIndependent =<< randn' [hidden_features, hidden_features]
      b_hh <- makeIndependent =<< randn' [hidden_features]
      return $ Recurrent w_ih b_ih w_hh b_hh


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

recurrent :: Recurrent -> Tensor -> Tensor -> Tensor
recurrent Recurrent{..} input hidden =
  Torch.Functions.tanh (ih + hh)
  where
    ih = (matmul input (toDependent weight_ih)) + (toDependent bias_ih)
    hh = (matmul hidden (toDependent weight_hh)) + (toDependent bias_hh)


