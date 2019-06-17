{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}

module Torch.NN where

import Control.Monad.State.Strict

import Torch.Autograd

type Parameter = IndependentTensor
type ParamStream a = State [Parameter] a

nextParameter :: ParamStream Parameter
nextParameter = do
  params <- get
  case params of
    [] -> error "Not enough parameters supplied to replaceParameters"
    (p : t) -> do put t; return p

class Parametrized f where
  flattenParameters :: f -> [Parameter]
  replaceOwnParameters :: f -> ParamStream f

replaceParameters :: Parametrized f => f -> [Parameter] -> f
replaceParameters f params =
  let (f', remaining) = runState (replaceOwnParameters f) params in
  if null remaining
    then f'
    else error "Some parameters in a call to replaceParameters haven't been consumed!"

class Randomizable spec f | spec -> f where
  sample :: spec -> IO f

class (Randomizable spec f, Parametrized f) => Module spec f
