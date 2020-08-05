{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}

module Model where

import Control.Monad (when)
import GHC.Generics
import Prelude hiding (exp, log)

import Dataset
import Torch

--
-- Model-generic utility functions
--

data (Optimizer o, Parameterized p) => OptimSpec o p = OptimSpec {
    optimizer :: o,
    batchSize :: Int,
    numIters :: Int,
    learningRate :: Tensor,
    lossFn :: p -> Tensor -> Tensor -> Tensor -- model, input, target
}

-- | Train a model
train
    :: (Dataset d, Optimizer o, Parameterized p, HasForward p Tensor Tensor)
    => OptimSpec o p -> d -> p -> IO p
train OptimSpec{..} dataset init = do
    trained <- foldLoop init numIters $
        \state iter -> do
            (input, label) <- getItem dataset (iter*batchSize) batchSize
            -- print $ shape input
            let loss = lossFn state input label

            let flatParameters = flattenParameters state
            let (Gradients gradients) = grad' loss flatParameters
            -- print $ sumAll <$> gradients

            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss learningRate
            pure $ replaceParameters state newParam
    pure trained

--
-- MLP Implementation
--

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
    } deriving (Show, Eq)

data MLP = MLP {
    mlpFC0 :: Linear,
    mlpFC1 :: Linear,
    mlpFC2 :: Linear
    } deriving (Generic, Show)

mlpTemp :: Float -> MLP -> Tensor -> Tensor
mlpTemp temperature MLP{..} input =
    logSoftmaxTemp (asTensor temperature)
    . linearForward mlpFC2
    . relu
    . linearForward mlpFC1
    . relu
    . linearForward mlpFC0
    $ input
  where
    logSoftmaxTemp t z = (z/t) - log (sumDim (Dim 1) KeepDim Float (exp (z/t)))

instance Parameterized MLP
instance HasForward MLP Tensor Tensor where
    forward = mlpTemp 1.0

instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

--
-- CNN Implementation
--

data CNNSpec = CNNSpec {
    conv0Spec :: Conv2dSpec,
    fc0Spec :: LinearSpec,
    fc1Spec :: LinearSpec
} deriving (Show, Eq)

data CNN = CNN {
    cnnConv0 :: Conv2d,
    cnnFC0 :: Linear,
    cnnFC1:: Linear
} deriving (Generic, Show)

instance Parameterized CNN
instance HasForward CNN Tensor Tensor where
    forward CNN {..} input =
        logSoftmax (Dim 0)
        . linearForward cnnFC1
        . relu
        . linearForward cnnFC0
        . reshape [batchSize, channels * (9 * 9)]
        -- . reshape [batchSize, -1]
        -- kernel stride padding dilation ceilMode
        . maxPool2d (3, 3) (3, 3) (0, 0) (1, 1) Floor
        . relu
        . conv2dForward cnnConv0 (1, 1) (2, 2)
        . reshape [batchSize, 1 , 28, 28]
        $ input
      where
        channels = (shape (toDependent . conv2dWeight $ cnnConv0)) !! 0
        batchSize = Prelude.div (product (shape input)) 784

instance Randomizable CNNSpec CNN where
    sample CNNSpec {..} = CNN
        <$> sample conv0Spec
        <*> sample fc0Spec
        <*> sample fc1Spec
