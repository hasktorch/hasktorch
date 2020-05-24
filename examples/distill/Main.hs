{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad (when)
import GHC.Generics
import Prelude hiding (exp, log)

import Torch
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V

import Distill
import Dataset

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

-- | Transform probabilities along one-hot-encoding dimensions into the digit value
maxIndex :: Tensor -> Tensor
maxIndex = Torch.argmax (Dim 1) RemoveDim

-- | Load MNIST data as dataset abstraction
loadMNIST dataLocation = do
    (train, test) <- V.initMnist dataLocation
    let mnistTrain = MNIST {
        dataset = train,
        idxList = V.randomIndexes (V.length train)
    }
    let mnistTest = MNIST {
        dataset = test,
        idxList = V.randomIndexes (V.length train)
    }
    pure (mnistTrain, mnistTest)

-- | Setup distillation parameters and run
runDistill :: (Dataset d) => d -> IO (MLP, MLP) 
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
        teacherView = \t inp -> ModelView (mlpTemp 20.0 t inp),
        studentView = \s inp -> ModelView (mlpTemp 1.0 s inp),
        distillLoss = \(ModelView tOutput) (ModelView sOutput) -> nllLoss' (maxIndex tOutput) sOutput
    }
    student <- distill distillSpec optimSpec mnistData
    pure (teacher, student)
  where
    teacherSpec = MLPSpec mnistDataDim 300 300 10
    studentSpec = MLPSpec mnistDataDim 30 30 10

main = do
    (mnistTrain, mnistTest) <- loadMNIST "datasets/mnist"
    (teacher :: MLP, student :: MLP) <- runDistill mnistTrain
    mapM (\idx -> do
        testImg <- V.getImages' 1 784 (dataset mnistTest) [idx]
        print $ shape testImg
        V.dispImage testImg
        putStrLn $ "Teacher      : " ++ (show . maxIndex $ forward teacher testImg)
        putStrLn $ "Student      : " ++ (show . maxIndex $ forward student testImg)
        putStrLn $ "Ground Truth : " ++ (show $ V.getLabels' 1 (dataset mnistTest) [idx])
        ) [0..10]
    putStrLn "Done"
