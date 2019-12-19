{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM)
import Data.List (foldl', scanl', intersperse)
import GHC.Generics
import Prelude hiding (exp)

import Torch.Tensor
import Torch.DType (DType (Float))
import Torch.TensorFactories (ones', rand', randn', randnLike)
import Torch.Functional hiding (linear)
import Torch.Autograd
import Torch.NN

-- Model Specification

data VAESpec = VAESpec {
  encoderSpec :: [LinearSpec],
  muSpec :: LinearSpec,
  logvarSpec :: LinearSpec,
  decoderSpec :: [LinearSpec],
  nonlinearitySpec :: Tensor -> Tensor
} deriving (Generic)

-- Model State

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

-- Output including latent mu and logvar used for VAE loss

data ModelOutput = ModelOutput {
  recon :: Tensor,
  mu :: Tensor,
  logvar :: Tensor
} deriving (Show)

-- Recon Error + KL Divergence VAE Loss
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

-- | MLP helper function for model used by both encoder & decoder
mlp :: [Linear] -> (Tensor -> Tensor) -> Tensor -> Tensor
mlp mlpState nonlin input = foldl' revApply input layerFunctionsList
  where
    layerFunctionsList = intersperse nonlin $ (map linear mlpState)
    revApply x f = f x

-- | Reparamaterization trick to sample from latent space while allowing differentiation
reparamaterize :: Tensor -> Tensor -> IO Tensor
reparamaterize mu logvar = do
    eps <- Torch.TensorFactories.randnLike mu
    pure $ mu + eps * exp (0.5 * logvar)

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: Tensor -> Int -> Int -> IO Tensor
mvnCholesky cov n axisDim = do
    samples <- randn' [axisDim, n]
    pure $ matmul l samples
    where
      l = cholesky cov Upper

main :: IO ()
main =
  let nSamples = 32768
      dataDim = 4
      hDim = 2
      zDim = 2
      batchSize = 256 -- TODO - crashes for case where any batch is of size n=1
      numIters = 8000
  in do
    init <- sample $ VAESpec {
        encoderSpec = [LinearSpec dataDim hDim],
        muSpec = LinearSpec hDim zDim,
        logvarSpec = LinearSpec hDim zDim,
        decoderSpec = [LinearSpec zDim hDim, LinearSpec hDim dataDim],
        nonlinearitySpec = relu }

    dat <- transpose2D <$>
      mvnCholesky (asTensor ([[1.0, 0.3, 0.1, 0.0],
                              [0.3, 1.0, 0.3, 0.1],
                              [0.1, 0.3, 1.0, 0.3],
                              [0.0, 0.1, 0.3, 1.0]] :: [[Float]]))
                              nSamples dataDim
    trained <- foldLoop init numIters $ \vaeState i -> do
      let startIndex = mod (batchSize * i) nSamples
          endIndex = Prelude.min (startIndex + batchSize) nSamples
          input = slice dat 0 startIndex endIndex 1 -- size should be [batchSize, dataDim]
      output <- model vaeState input
      let (reconX, muVal, logvarVal) = (squeezeAll $ recon output, mu output, logvar output )
      let loss = vaeLoss reconX input muVal logvarVal
      let flat_parameters = flattenParameters vaeState
      let gradients = grad loss flat_parameters
      if i `mod` 100 == 0
          then do putStrLn $ show loss
          else return ()

      new_flat_parameters <- mapM makeIndependent $ sgd 1e-6 flat_parameters gradients
      pure $ replaceParameters vaeState $ new_flat_parameters
    putStrLn "Done"
  where
    foldLoop x count block = foldM block x [0..count]
