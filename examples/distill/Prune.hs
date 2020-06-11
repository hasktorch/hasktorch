{-# LANGUAGE RecordWildCards #-}

module Main where

import Dataset
import Torch
import Distill
import Model

data (Parameterized m) => PruneSpec m = PruneSpec {
    ref :: m,
    getWeights :: m -> [Tensor],
    pruneWeights :: Float -> Tensor -> Tensor
}


-- | Setup pruning parameters and run
runPrune :: (Dataset d) => d -> IO (MLP, MLP) 
runPrune mnistData = do
    -- Train teacher
    initRef <- sample refSpec
    let optimSpec = OptimSpec {
        optimizer = GD,
        batchSize = 256,
        numIters = 500,
        learningRate = 1e-3
    }
    ref <- train optimSpec mnistData initRef

    -- Prune to student
    -- target <- sample studentSpec
    let pruneSpec = PruneSpec {
        ref = ref,
        getWeights = \teacher -> toDependent <$> flattenParameters teacher
    }
    -- student <- distill distillSpec optimSpec mnistData
    let pruned = undefined
    pure (ref, pruned)
  where
    refSpec = MLPSpec mnistDataDim 300 300 10

main = do
    putStrLn "Done"
