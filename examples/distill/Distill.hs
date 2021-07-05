{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}

module Distill where

import qualified Control.Monad.State as S
import Control.Monad (when)
import GHC.Generics
import Prelude hiding (exp, log)

import Torch
import Dataset
import Model (OptimSpec(..))

newtype ModelView = ModelView { view :: Tensor }

data DistillSpec t s where
  DistillSpec :: (Parameterized s, Parameterized t) => {
    teacher :: t,
    student :: s,
    teacherView :: t -> Tensor -> ModelView,
    studentView :: s -> Tensor -> ModelView,
    distillLoss :: ModelView -> ModelView -> Tensor
    } -> DistillSpec t s

-- | Distill a teacher to a student
distill
    :: (Optimizer o, MockDataset d)
    => DistillSpec t s
    -> OptimSpec o s
    -> d
    -> IO s
distill DistillSpec{..} OptimSpec{..} dataset = do
    trained <- foldLoop student numIters $
        \state iter -> do
            (input, _) <- Dataset.getItem dataset (iter * batchSize) batchSize
            let tOutput = teacherView teacher input
                sOutput = studentView state input
                loss = distillLoss tOutput sOutput
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss learningRate
            pure newParam
    pure trained
