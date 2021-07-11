{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad (when)
import Dataset
import Distill
import GHC.Generics
import Model
import Torch
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V

-- | Transform probabilities along one-hot-encoding dimensions into the digit value
maxIndex :: Tensor -> Tensor
maxIndex = Torch.argmax (Dim 1) RemoveDim

-- | Setup distillation parameters and run
runDistill :: (MockDataset d) => d -> IO (MLP, MLP)
runDistill mnistData = do
  -- Train teacher
  initTeacher <- sample teacherSpec
  let optimSpec =
        OptimSpec
          { optimizer = GD,
            batchSize = 256,
            numIters = 500,
            learningRate = 1e-3,
            lossFn = \model input target -> nllLoss' target (forward model input)
          }
  teacher <- train optimSpec mnistData initTeacher
  -- Distill student
  initStudent <- sample studentSpec
  let distillSpec =
        DistillSpec
          { teacher = teacher,
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
  mapM
    ( \idx -> do
        let testImg = V.getImages' 1 784 (dataset mnistTest) [idx]
        print $ shape testImg
        V.dispImage testImg
        putStrLn $ "Teacher      : " ++ (show . maxIndex $ forward teacher testImg)
        putStrLn $ "Student      : " ++ (show . maxIndex $ forward student testImg)
        putStrLn $ "Ground Truth : " ++ (show $ V.getLabels' 1 (dataset mnistTest) [idx])
    )
    [0 .. 10]
  putStrLn "Done"
