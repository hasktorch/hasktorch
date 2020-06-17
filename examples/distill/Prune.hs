{-# LANGUAGE RecordWildCards #-}

module Main where

import Dataset
import Torch
import Model

data (Parameterized m) => PruneSpec m = PruneSpec {
    ref :: m,
    selectWeights :: m -> [Tensor],
    pruneWeights :: Float -> Tensor -> Tensor
}

selectAllWeights model = 
    toDependent <$> flattenParameters model

l1Prune :: Float -> Tensor -> Tensor
l1Prune threshold t =
    Torch.abs t `lt` (asTensor threshold)

-- | Setup pruning parameters and run
runPrune :: (Dataset d) => d -> IO (CNN, CNN) 
runPrune mnistData = do

    print "sampling"
    -- train parent model
    initRef <- sample refSpec
    let optimSpec = OptimSpec {
        optimizer = GD,
        batchSize = 256,
        numIters = 100,
        learningRate = 1e-6, 
        lossFn = nllLoss' 
    }
    print "training"
    ref <- train optimSpec mnistData initRef

    -- l1 test
        {-
    l1 <- train 
        -- TODO XXX = weights
        optimSpec {
            lossFn = \t t' -> nllLoss' t t' + 1.0 * l1Loss ReduceSum XXX zerosLike XXX 
            }
        mnistData 
        initRef
        -}

    -- prune target
    let pruneSpec = PruneSpec {
        ref = ref,
        selectWeights = selectAllWeights,
        pruneWeights = l1Prune
    }
    print "pruning"
    -- student <- distill distillSpec optimSpec mnistData
    let pruned = undefined
    pure (ref, pruned)
  where
    channels = 4
    refSpec = 
        CNNSpec
            -- input channels, output channels, kernel height, kernel width
            (Conv2dSpec 1 channels 5 5)
            -- (LinearSpec (784*channels) 100)
            (LinearSpec (9*9*channels) 100)

main = do
    putStrLn "Dim Check"
    print $ maxPool2dDim (3, 3) (3, 3) (0, 0) (1, 1) FloorMode (28, 28)
    print $ maxPool2dDim (3, 3) (6, 3) (0, 0) (1, 1) FloorMode (28, 28)
    print $ maxPool2dDim (3, 3) (6, 6) (0, 0) (1, 1) FloorMode (28, 28)
    putStrLn "Loading Data"
    (mnistTrain, mnistTest) <- loadMNIST "datasets/mnist"
    putStrLn "Running Prune"
    (original, derived) <- runPrune mnistTrain
    putStrLn "Done"
