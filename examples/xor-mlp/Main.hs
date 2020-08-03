{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import GHC.Generics

import Control.Monad (when)
import Data.List (foldl', scanl', intersperse)

import Torch

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

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
-- This instance automatically generates the following code
-----------------------------------------------------------
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

batchSize = 2
numIters = 2000

model :: MLP -> Tensor -> Tensor
model params t = mlp params t

main :: IO ()
main = do
    init <- sample $ MLPSpec { feature_counts = [2, 2, 1], 
                               nonlinearitySpec = Torch.tanh } 
    trained <- foldLoop init numIters $ \state i -> do
        input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
        let (y, y') = (tensorXOR input, squeezeAll $ model state input)
            loss = mseLoss y y'
        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
        (newParam, _) <- runStep state optimizer loss 1e-1
        return $ replaceParameters state $ newParam
    putStrLn "Final Model:"
    putStrLn $ "0, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 0 :: Float]))
    putStrLn $ "0, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 1 :: Float]))
    putStrLn $ "1, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 0 :: Float]))
    putStrLn $ "1, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 1 :: Float]))
    return ()
  where
    optimizer = GD
    tensorXOR :: Tensor -> Tensor
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t
