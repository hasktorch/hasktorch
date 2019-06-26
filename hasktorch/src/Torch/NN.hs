{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.NN where

import Control.Monad.State.Strict

import Torch.Autograd
import Torch.Tensor
import Torch.TensorFactories (ones', rand', randn')
import Torch.Functions

type Parameter = IndependentTensor
type ParamStream a = State [Parameter] a

nextParameter :: ParamStream Parameter
nextParameter = do
  params <- get
  case params of
    [] -> error "Not enough parameters supplied to replaceParameters"
    (p : t) -> do put t; return p

class Parameterized f where
  flattenParameters :: f -> [Parameter]
  replaceOwnParameters :: f -> ParamStream f

replaceParameters :: Parameterized f => f -> [Parameter] -> f
replaceParameters f params =
  let (f', remaining) = runState (replaceOwnParameters f) params in
  if null remaining
    then f'
    else error "Some parameters in a call to replaceParameters haven't been consumed!"

class Randomizable spec f | spec -> f where
  sample :: spec -> IO f

class (Randomizable spec f, Parameterized f) => Module spec f

data LinearSpec = LinearSpec { in_features :: Int, out_features :: Int }
  deriving (Show, Eq)

data Linear = Linear { weight :: Parameter, bias :: Parameter } deriving Show

instance Randomizable LinearSpec Linear where
  sample LinearSpec{..} = do
      w <- makeIndependent =<< randn' [in_features, out_features]
      b <- makeIndependent =<< randn' [out_features]
      return $ Linear w b

instance Parameterized Linear where
  flattenParameters Linear{..} = [weight, bias]
  replaceOwnParameters _ = do
    weight <- nextParameter
    bias <- nextParameter
    return $ Linear{..}

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith step depParameters gradients
  where 
    step p dp = p - (lr * dp)
    depParameters = (map toDependent parameters)
