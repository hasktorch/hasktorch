{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (when)
import GHC.Generics
import Prelude hiding (exp, log)

import Torch
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V
import Distill

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

instance Parameterized MLP where
    forward = mlpTemp 1.0

instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP 
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

train :: Optimizer o => OptimSpec o -> MNIST -> MLP -> IO MLP
train OptimSpec{..} dataset init = do
    trained <- foldLoop init numIters $
        \state iter -> do
            ((input, label), _) <- get dataset batchSize
            let loss = nllLoss' label $ forward state input
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss learningRate
            pure $ replaceParameters state newParam
    pure trained

maxIndex = Torch.argmax (Dim 1) RemoveDim

runDistill :: MNIST -> IO (MLP, MLP) 
runDistill mnistData = do
    -- Train teacher
    initTeacher <- sample teacherSpec
    let optimSpec = OptimSpec {
        optimizer = GD,
        batchSize = 256,
        numIters = 500,
        learningRate = 1e-3
    }
    teacher <- train optimSpec mnistData initTeacher
    -- Distill student
    initStudent <- sample studentSpec
    let distillSpec = DistillSpec {
        teacher = teacher,
        student = initStudent,
        teacherLens = mlpTemp 20.0,
        studentLens = mlpTemp 1.0,
        distillLoss = \tOutput sOutput -> nllLoss' (maxIndex tOutput) sOutput
    }
    student <- distill distillSpec optimSpec mnistData
    pure (teacher, student)
  where
    teacherSpec = MLPSpec dataDim 300 300 10
    studentSpec = MLPSpec dataDim 30 30 10

main = do

    (trainData, testData) <- V.initMnist "datasets/mnist"
    let mnist = MNIST {
        trainData = trainData,
        testData = testData,
        idxList = V.randomIndexes (V.length trainData),
        index = 0
    }
    (teacher, student) <- runDistill mnist
    mapM (\idx -> do
        testImg <- V.getImages' 1 784 testData [idx]
        print $ shape testImg
        V.dispImage testImg
        putStrLn $ "Teacher      : " ++ (show . maxIndex $ forward teacher testImg)
        putStrLn $ "Student      : " ++ (show . maxIndex $ forward student testImg)
        putStrLn $ "Ground Truth : " ++ (show $ V.getLabels' 1 testData [idx])
        ) [0..10]
    putStrLn "Done"
