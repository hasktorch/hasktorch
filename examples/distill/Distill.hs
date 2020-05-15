{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Distill where

import Control.Monad (when)
import GHC.Generics
import Prelude hiding (exp, log)

import Torch
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V

-- hard coded placeholder for this example until we have a more generic dataset types
type Dataset = V.MnistData 
dataDim = 784 :: Int

data Parameterized p => DistillSpec p = DistillSpec {
    teacher :: p,
    student :: p,
    teacherLens :: p -> Tensor -> Tensor,
    studentLens :: p -> Tensor -> Tensor,
    distillLoss :: Tensor -> Tensor -> Tensor
}

data Optimizer o => OptimSpec o = OptimSpec {
    optimizer :: o,
    batchSize :: Int,
    numIters :: Int,
    learningRate :: Tensor
}

distill
    :: (Parameterized p, Optimizer o)
    => DistillSpec p
    -> OptimSpec o
    -> Dataset
    -> IO p 
distill DistillSpec{..} OptimSpec{..} trainData = do
    let nImages = V.length trainData
        idxList = V.randomIndexes nImages
    trained <- foldLoop student numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            input <- V.getImages' batchSize dataDim trainData idx
            let tOutput = teacherLens teacher input
                sOutput = studentLens state input
                loss = distillLoss tOutput sOutput
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss learningRate
            pure $ replaceParameters state newParam
    pure trained

