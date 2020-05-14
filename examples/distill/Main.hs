{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (when)
import GHC.Generics
import Prelude hiding (exp, log)
import System.Random (mkStdGen, randoms)

import Torch
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V

dataDim = 784
batchSize = 256
numIters = 500
teacherSpec = MLPSpec dataDim 300 300 10
studentSpec = MLPSpec dataDim 30 30 10

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
    } deriving (Show, Eq)

data MLP = MLP { 
    l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
    } deriving (Generic, Show)

instance Parameterized MLP
instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP 
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlpTemp :: Float -> MLP -> Tensor -> Tensor
mlpTemp temperature MLP{..} input = 
    logSoftmaxTemp (asTensor temperature)
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input
    where
        logSoftmaxTemp t z = (z/t) - log (sumDim (Dim 1) KeepDim Float (exp (z/t)))

mlp = mlpTemp 1.0

train :: V.MnistData -> MLP -> IO MLP
train trainData init = do
    let optimizer = GD
    let nImages = V.length trainData
        idxList = V.randomIndexes nImages
    trained <- foldLoop init numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            input <- V.getImages' batchSize dataDim trainData idx
            let label = V.getLabels' batchSize trainData idx
                loss = nllLoss' label $ mlp state input
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss 1e-3 -- GD
            pure $ replaceParameters state newParam
    pure trained

maxIndex = Torch.argmax (Dim 1) RemoveDim

distill :: V.MnistData -> MLP -> MLP -> IO MLP
distill trainData teacher studentInit = do
    let optimizer = GD
    let nImages = V.length trainData
        idxList = V.randomIndexes nImages
    trained <- foldLoop studentInit numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            input <- V.getImages' batchSize dataDim trainData idx
            let tOutput = mlpTemp 20.0 teacher input
                sOutput = mlp state input
            let label = maxIndex tOutput
            let loss = nllLoss' label sOutput
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss 1e-3 -- GD
            pure $ replaceParameters state newParam
    pure trained

main = do
    (trainData, testData) <- V.initMnist "datasets/mnist"
    initTeacher <- sample teacherSpec
    initStudent <- sample studentSpec
    teacher <- train trainData initTeacher
    student <- distill trainData teacher initStudent
    mapM (\idx -> do
        testImg <- V.getImages' 1 784 testData [idx]
        print $ shape testImg
        V.dispImage testImg
        putStrLn $ "Teacher      : " ++ (show . maxIndex $ mlp teacher testImg)
        putStrLn $ "Student      : " ++ (show . maxIndex $ mlp student testImg)
        putStrLn $ "Ground Truth : " ++ (show $ V.getLabels' 1 testData [idx])
        ) [0..10]
    putStrLn "Done"
