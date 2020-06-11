{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Distill where

import qualified Control.Monad.State as S
import Control.Monad (when)
import GHC.Generics
import Prelude hiding (exp, log)

import Torch
import Dataset

newtype ModelView = ModelView { view :: Tensor }

data (Parameterized t, Parameterized s) => DistillSpec t s = DistillSpec {
    teacher :: t,
    student :: s,
    teacherView :: t -> Tensor -> ModelView,
    studentView :: s -> Tensor -> ModelView,
    distillLoss :: ModelView -> ModelView -> Tensor
}

data Optimizer o => OptimSpec o = OptimSpec {
    optimizer :: o,
    batchSize :: Int,
    numIters :: Int,
    learningRate :: Tensor
}

-- | Train the teacher model
train :: (Dataset d, Optimizer o, Parameterized p, HasForward p Tensor Tensor) => OptimSpec o -> d -> p -> IO p
train OptimSpec{..} dataset init = do
    trained <- foldLoop init numIters $
        \state iter -> do
            (input, label) <- getItem dataset (iter*batchSize) batchSize
            let loss = nllLoss' label $ forward state input
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss learningRate
            pure $ replaceParameters state newParam
    pure trained

-- | Distill a teacher to a student
distill
    :: (Parameterized t, Parameterized s, Optimizer o, Dataset d)
    => DistillSpec t s
    -> OptimSpec o
    -> d
    -> IO s
distill DistillSpec{..} OptimSpec{..} dataset = do
    trained <- foldLoop student numIters $
        \state iter -> do
            (input, _) <- getItem dataset (iter * batchSize) batchSize
            let tOutput = teacherView teacher input
                sOutput = studentView state input
                loss = distillLoss tOutput sOutput
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss learningRate
            pure $ replaceParameters state newParam
    pure trained

