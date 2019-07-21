{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM)
import Prelude hiding (exp)

import Torch.Tensor
import Torch.DType (DType (Float))
import Torch.TensorFactories (ones', rand', randn')
import Torch.Functions
import Torch.Autograd
import Torch.NN

import GHC.Generics

data VAESpec = VAESpec {
  encoderSpec :: [LinearSpec],
  decoderSpec :: [LinearSpec]
} deriving (Show, Generic)

data VAEState = VAEState {
  encoderState :: [Linear],
  decoderState :: [Linear]
} deriving (Show, Generic)

instance Randomizable VAESpec VAEState where
  sample VAESpec{..} = do
    pure undefined
    
instance Parameterized VAEState

reparamaterize :: Tensor -> Tensor -> IO Tensor
reparamaterize mu logvar = do
    eps <- undefined -- eps = torch.randn_like(std)
    pure undefined --  mu + eps * std
  where 
      std = exp (0.5 * logvar)

      
linear Linear{..} input = squeezeAll $ matmul input depWeight + depBias
  where (depWeight, depBias) = (toDependent weight, toDependent bias)

model :: VAEState -> Tensor -> Tensor
model VAEState{..} input = undefined

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: Tensor -> Int -> Int -> IO Tensor
mvnCholesky cov n axisDim = do
    samples <- randn' [axisDim, n]
    pure $ matmul l samples
    where 
      l = cholesky cov Upper

main = do
    putStrLn "Done"
