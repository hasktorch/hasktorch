{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module LinearLayer where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Parameters


data LinearSpec = LinearSpec { in_features :: Int, out_features :: Int }
  deriving (Show, Eq)

data Linear = Linear { weight :: Parameter, bias :: Parameter }
  deriving (Show)

instance Randomizable LinearSpec Linear where
  sample LinearSpec{..} = do
      w <- makeIndependent =<< randn' [in_features, out_features]
      b <- makeIndependent =<< randn' [out_features]
      return $ Linear w b

instance Parametrized Linear where
  flattenParameters Linear{..} = [weight, bias]
  replaceOwnParameters _ = do
    weight <- nextParameter
    bias <- nextParameter
    return $ Linear{..}


linear :: Linear -> Tensor -> Tensor
linear Linear{..} input = (matmul input (toDependent weight)) + (toDependent bias)
