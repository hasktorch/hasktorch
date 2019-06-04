{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

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

--------------------------------------------------------------------------------
-- Linear function
--------------------------------------------------------------------------------

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

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec { feature_counts :: [Int], nonlinearitySpec :: Tensor -> Tensor }

data MLP = MLP { layers :: [Linear], nonlinearity :: Tensor -> Tensor }

instance Randomizable MLPSpec MLP where
  sample MLPSpec{..} = do
      let layer_sizes = mkLayerSizes feature_counts
      linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
      return $ MLP { layers = linears, nonlinearity = nonlinearitySpec }
    where
      mkLayerSizes (a : (b : t)) =
          scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

instance Parametrized MLP where
  flattenParameters MLP{..} = concat $ map flattenParameters layers
  replaceOwnParameters mlp = do
    new_layers <- mapM replaceOwnParameters (layers mlp)
    return $ mlp { layers = new_layers }

mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where revApply x f = f x

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batch_size = 32
num_iters = 10000

model :: MLP -> Tensor -> Tensor
model params t = sigmoid (mlp params t)

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith (\p dp -> p - (lr * dp)) (map toDependent parameters) gradients

main :: IO ()
main = do
    init <- sample $ MLPSpec { feature_counts = [2, 20, 20, 1], nonlinearitySpec = Torch.Functions.tanh }
    trained <- foldLoop init num_iters $ \state i -> do
        input <- rand' [batch_size, 2] >>= return . (toDType Float) . (gt 0.5)
        let expected_output = tensorXOR input

        let output = squeezeAll $ model state input
        let loss = mse_loss output expected_output

        let flat_parameters = flattenParameters state
        let gradients = grad loss flat_parameters

        if i `mod` 100 == 0
          then do putStrLn $ show loss
          else return ()

        new_flat_parameters <- mapM makeIndependent $ sgd 5e-4 flat_parameters gradients
        return $ replaceParameters state $ new_flat_parameters
    return ()
  where
    foldLoop x count block = foldM block x [1..count]

    tensorXOR :: Tensor -> Tensor
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select t 1 0
        b = select t 1 1
