{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import System.Process (system)
import Dataset
import Graphics.Vega.VegaLite hiding (sample, shape)
import Plot
import Torch
import Model

data (Parameterized m) => PruneSpec m = PruneSpec {
    selectWeights :: m -> [Tensor],
    pruneWeights :: m -> m
}

l1 :: Tensor -> Tensor
l1 w = l1Loss ReduceMean w (zerosLike w)

l2 :: Tensor -> Tensor
l2 x = mseLoss (zerosLike x) x

mkReg regFn selectFn a b model input target =
    a * (nllLoss' target (forward model input)) + b * regFn selectParams
    where
        selectParams = flattenAll $ cat (Dim 0) $ flattenAll <$> (selectFn $ model) 

-- | Setup pruning parameters and run
regularizationTest :: (Dataset d) => d -> IO () 
regularizationTest mnistData = do

    print "sampling"
    -- train reference model
    initRef <- sample refSpec
    let optimSpec = OptimSpec {
        optimizer = mkAdam 0 0.9 0.999 (flattenParameters initRef),
        batchSize = 128,
        numIters = 500,
        learningRate = 5e-5, 
        lossFn = \model input target -> nllLoss' target (forward model input)
    } :: OptimSpec Adam CNN

    print "training"
    ref <- train optimSpec mnistData =<< sample refSpec

    let pruneSpec = PruneSpec {
        selectWeights = \m -> [toDependent . weight . cnnFC0 $ m,
                               toDependent . weight . cnnFC1 $ m],
        pruneWeights = undefined
    }

            
    print "l1"
    initRefL1 <- sample refSpec

    l1Model <- train
            optimSpec { optimizer = mkAdam (0 :: Int) 0.9 0.999 (flattenParameters initRefL1),  
                        lossFn = mkReg l1 (selectWeights pruneSpec) 1.0 10.0 }
            mnistData 
            initRefL1

    print "l2"
    initRefL2 <- sample refSpec
    l2Model <- train
        optimSpec { optimizer = mkAdam 0 0.9 0.999 (flattenParameters initRefL2), 
                    lossFn = mkReg l2 (selectWeights pruneSpec) 1.0 1.0 }
        mnistData 
        initRefL2

    plt <- strip (toDependent . weight . cnnFC0 $ initRef)
    toHtmlFile "plotInit.html" plt
    system "open plotInit.html"
    print "weights0 l1"
    plt <- strip (toDependent . weight . cnnFC0 $ l1Model)
    toHtmlFile "plot0l1.html" plt
    system "open plot0l1.html"
    print "weights0 l2"
    plt <- strip (toDependent . weight . cnnFC0 $ l2Model)
    toHtmlFile "plot0l2.html" plt
    system "open plot0l2.html"

    -- TODO - structured/unstructured pruning test

    pure ()
  where
    channels = 3
    refSpec = 
        CNNSpec
            -- input channels, output channels, kernel height, kernel width
            (Conv2dSpec 1 channels 5 5)
            -- (no ADT : maxPool2d (3, 3) (3, 3) (0, 0) (1, 1) Floor )
            (LinearSpec (9*9*channels) 80)
            (LinearSpec 80 10)
    

pruneTest = do
    putStrLn "Loading Data"
    (mnistTrain, mnistTest) <- loadMNIST "datasets/mnist"
    regularizationTest mnistTrain
    putStrLn "Done"

main = pruneTest