{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Arrow ((>>>))
import Control.Monad (foldM)
import Data.List (foldl', scanl', intersperse)
import GHC.Generics
import Prelude hiding (exp)

import Torch.Tensor
import Torch.DType (DType (Float))
import Torch.TensorFactories (ones', rand', randn', randn_like)
import Torch.Functions
import Torch.Autograd
import Torch.NN

data VAESpec = VAESpec {
  encoderSpec :: [LinearSpec],
  muSpec :: LinearSpec,
  logvarSpec :: LinearSpec,
  decoderSpec :: [LinearSpec],
  nonlinearitySpec :: Tensor -> Tensor
} deriving (Generic)

data VAEState = VAEState {
  encoderState :: [Linear],
  muFC :: Linear,
  logvarFC :: Linear,
  decoderState :: [Linear],
  nonlinearity :: Tensor -> Tensor
} deriving (Generic)

instance Randomizable VAESpec VAEState where
  sample VAESpec{..} = do
    encoderState <- mapM sample encoderSpec
    muFC <- sample muSpec
    logvarFC <- sample logvarSpec
    decoderState <- mapM sample decoderSpec
    let nonlinearity = nonlinearitySpec
    pure $ VAEState{..}
    
instance Parameterized VAEState

data ModelOutput = ModelOutput {
  recon :: Tensor,
  mu :: Tensor,
  logvar :: Tensor
} deriving (Show)

vaeLoss :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
vaeLoss recon_x x mu logvar = reconLoss + kld
  where
    -- reconLoss = binary_cross_entropy_loss recon_x x undefined ReduceSum
    reconLoss = mse_loss recon_x x
    kld = -0.5 * (sumAll (1 + logvar - pow mu (2 :: Int) - exp logvar))

-- | End-to-end function for VAE model
model :: VAEState -> Tensor -> IO ModelOutput
model VAEState{..} input = do
    let encoded = mlp encoderState nonlinearity input
        mu = (linear muFC) encoded
        logvar = (linear logvarFC) encoded
    z <- reparamaterize mu logvar
    let output = mlp decoderState nonlinearity z
    pure $ ModelOutput output mu logvar

-- | MLP helper function used for encoder & decoder
mlp :: [Linear] -> (Tensor -> Tensor) -> Tensor -> Tensor
mlp mlpState nonlin input = foldl' revApply input layerFunctionsList
  where 
    layerFunctionsList = intersperse nonlin $ (map linear mlpState) 
    revApply x f = f x

-- | Reparamaterization trick to sample from latent space while allowing differentiation
reparamaterize :: Tensor -> Tensor -> IO Tensor
reparamaterize mu logvar = do
    eps <- randn_like mu
    pure $ mu + eps * exp (0.5 * logvar)
      
-- | Given weights, apply linear layer to an input
linear :: Linear -> Tensor -> Tensor
linear Linear{..} input = squeezeAll $ matmul input depWeight + depBias
  where (depWeight, depBias) = (toDependent weight, toDependent bias)

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: Tensor -> Int -> Int -> IO Tensor
mvnCholesky cov n axisDim = do
    samples <- randn' [axisDim, n]
    pure $ matmul l samples
    where 
      l = cholesky cov Upper

main = do
    init <- sample $ VAESpec {
        encoderSpec = [LinearSpec dataDim 15],
        muSpec = LinearSpec 15 3,
        logvarSpec = LinearSpec 15 3,
        decoderSpec = [LinearSpec 3 15, LinearSpec 15 dataDim],
        nonlinearitySpec = relu }

    dat <- mvnCholesky (asTensor ([[1, 0.3], [0.3, 1.0]] :: [[Float]])) 1000 dataDim

    trained <- foldLoop init num_iters $ \vaeState i -> do
      input <- randn' [batchSize, dataDim]
      output <- model vaeState input
      let reconX = squeezeAll $ recon output
      let (muVal, logvarVal) = (mu output, logvar output)
      let loss = vaeLoss reconX input muVal logvarVal
      let flat_parameters = flattenParameters vaeState
      let gradients = grad loss flat_parameters
      if i `mod` 100 == 0
          then do putStrLn $ show loss
          else return ()

      new_flat_parameters <- mapM makeIndependent $ sgd 5e-4 flat_parameters gradients
      pure $ replaceParameters vaeState $ new_flat_parameters
    putStrLn "Done"
    pure init
  where
    dataDim = 2
    batchSize = 256
    num_iters = 10000
    foldLoop x count block = foldM block x [1..count]
