{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions hiding (linear)
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN
import GHC.Generics

import Control.Monad (foldM)
import Data.List (foldl', scanl', intersperse)

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

linear :: Linear -> Tensor -> Tensor
linear Linear{..} input = (matmul input (toDependent weight)) + (toDependent bias)

data MLPSpec = MLPSpec { feature_counts :: [Int], nonlinearitySpec :: Tensor -> Tensor }

data MLP = MLP { layers :: [Linear], nonlinearity :: Tensor -> Tensor } deriving (Generic)

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

instance Parameterized MLP
-- This instance generates the following code
---------------------------------------------------
-- instance Parameterized MLP where
--   flattenParameters MLP{..} = concat $ map flattenParameters layers
--   replaceOwnParameters mlp = do
--     new_layers <- mapM replaceOwnParameters (layers mlp)
--     return $ mlp { layers = new_layers }

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

main :: IO ()
main = do
    init <- sample $ MLPSpec { feature_counts = [2, 20, 20, 1], 
                               nonlinearitySpec = Torch.Functions.tanh }
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
