{-# LANGUAGE RecordWildCards #-}

module Main where

import System.Cmd (system)
import Dataset
import Graphics.Vega.VegaLite hiding (sample, shape)
import Plot (histogram, scatter)
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
    -- train reference model
    initRef <- sample refSpec
    let optimSpec = OptimSpec {
        optimizer = GD,
        batchSize = 256,
        numIters = 50,
        learningRate = 1e-6, 
        lossFn = nllLoss' 
    }
    print "training"
    ref <- train optimSpec mnistData initRef

    let pruneSpec = PruneSpec {
        ref = initRef,
        selectWeights = (:[]) . toDependent . weight . cnnFC0 
    }

    -- l1
    l1 <- train 
        optimSpec {
            numIters = 100,
            lossFn = \t t' -> 
                let regWeights = head (selectWeights pruneSpec $ initRef) in
                    let regTerm = l1Loss ReduceSum regWeights (zerosLike regWeights) in
                        nllLoss' t t' + 1 * regTerm 
            }
        mnistData 
        initRef

    print "weights ref"
    plt <- histogram (toDependent . weight . cnnFC0 $ ref)
    toHtmlFile "plot0.html" plt
    system "open plot0.html"

    print "weights l1"
    plt <- histogram (toDependent . weight . cnnFC0 $ l1)
    toHtmlFile "plot1.html" plt
    system "open plot1.html"

    let resultWt = head $ (selectWeights pruneSpec) l1 
    -- print $ resultWt
    print $ shape resultWt

    print "pruning"
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
    print $ maxPool2dDim (3, 3) (3, 3) (0, 0) (1, 1) Floor (28, 28)
    print $ maxPool2dDim (3, 3) (6, 3) (0, 0) (1, 1) Floor (28, 28)
    print $ maxPool2dDim (3, 3) (6, 6) (0, 0) (1, 1) Floor (28, 28)
    putStrLn "Loading Data"
    (mnistTrain, mnistTest) <- loadMNIST "datasets/mnist"
    putStrLn "Running Prune"
    (original, derived) <- runPrune mnistTrain
    -- scatter (asTensor [1 :: Float, 2, 3]) (asTensor [5 :: Float, 2, 3])
    -- toHtmlFile "plot.html" <$> histogram (asTensor [5 :: Float, 2, 3]) 
    -- system "open plot.html"
    putStrLn "Done"
