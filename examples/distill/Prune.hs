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

l1 :: Tensor -> Tensor
l1 w = l1Loss ReduceSum w (zerosLike w)

-- | Setup pruning parameters and run
runPrune :: (Dataset d) => d -> IO (CNN, CNN) 
runPrune mnistData = do

    print "sampling"
    -- train reference model
    initRef <- sample refSpec
    let optimSpec = OptimSpec {
        optimizer = GD,
        batchSize = 256,
        numIters = 200,
        learningRate = 1e-6, 
        lossFn = nllLoss' 
    }
    print "training"
    ref <- train optimSpec mnistData initRef

    let pruneSpec = PruneSpec {
        ref = initRef,
        -- selectWeights = (:[]) . toDependent . weight . cnnFC0 
        selectWeights = \m -> [toDependent . weight . cnnFC0 $ m,
                               toDependent . weight . cnnFC1 $ m]
    }

    -- l1
    l1Model <- train 
        optimSpec {
            -- numIters = 1000,
            lossFn = \t t' -> 
                let regWeights = head (selectWeights pruneSpec $ initRef) in
                     nllLoss' t t' + 0.01 * l1 regWeights 
            }
        mnistData 
        initRef

    print "weights0 ref"
    plt <- histogram (toDependent . weight . cnnFC0 $ ref)
    toHtmlFile "plot0.html" plt
    system "open plot0.html"

    print "weights1 ref"
    plt <- histogram (toDependent . weight . cnnFC1 $ ref)
    toHtmlFile "plot0.html" plt
    system "open plot0.html"

    print "weights0 l1"
    plt <- histogram (toDependent . weight . cnnFC0 $ l1Model)
    toHtmlFile "plot0l1.html" plt
    system "open plot0l1.html"

    print "weights1 l1"
    plt <- histogram (toDependent . weight . cnnFC1 $ l1Model)
    toHtmlFile "plot1l1.html" plt
    system "open plot1l1.html"

    let resultWt = head $ (selectWeights pruneSpec) l1Model
    -- print $ resultWt
    print $ shape resultWt

    print "pruning"
    let pruned = undefined
    pure (ref, pruned)
  where
    channels = 3
    refSpec = 
        CNNSpec
            -- input channels, output channels, kernel height, kernel width
            (Conv2dSpec 1 channels 5 5)
            -- (no ADT : maxPool2d (3, 3) (3, 3) (0, 0) (1, 1) Floor )
            (LinearSpec (9*9*channels) 100)
            (LinearSpec 100 30)

main = do
    putStrLn "Loading Data"
    (mnistTrain, mnistTest) <- loadMNIST "datasets/mnist"
    putStrLn "Running Prune"
    (original, derived) <- runPrune mnistTrain
    putStrLn "Done"
