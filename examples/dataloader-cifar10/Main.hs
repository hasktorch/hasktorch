{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (when)
import GHC.Generics
import System.Random (mkStdGen, randoms)
import Prelude hiding (exp)

import Torch.Autograd as A
import Torch.Functional hiding (take)
import Torch.Tensor
import Torch.NN
import Numeric.Dataloader
import Numeric.Datasets
import Numeric.Datasets.CIFAR10

import Common (foldLoop)

data CNNSpec = CNNSpec
data MaxPool 
data Conv2d

data CNN = CNN { 
    l10 :: Conv2d,
    l11 :: MaxPool,
    l20 :: Conv2d,
    l21 :: MaxPool,
    l31 :: Linear,
    l32 :: Linear,
    l33 :: Linear
    } deriving (Generic, Show)

instance Parameterized CNN
instance Randomizable CNNSpec CNN where
    sample CNNSpec {..} = undefined
      -- CNN
      --   <$> sample (LinearSpec inputFeatures hiddenFeatures0)
      --   <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
      --   <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

cnn :: CNN -> Tensor -> Tensor
cnn CNN{..} input = undefined
    -- . linear l33
    -- . relu
    -- . linear l32
    -- . relu
    -- . linear l31
    -- . reshape [-1,16*5*5]
    -- . maxpool l21
    -- . relu
    -- . conv2d l20
    -- . maxpool l11
    -- . relu
    -- . conv2d l10
    -- $ input

train :: Stream (Of [b]) IO () -> IO CNN
train trainData = do
    init <- sample spec
    let nImages = I.length trainData
        idxList = randomIndexes nImages
    trained <- foldLoop init numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            input <- UI.getImages' batchSize dataDim trainData idx
            let label = UI.getLabels' batchSize trainData idx
                loss = nllLoss' (cnn state input) label
                flatParameters = flattenParameters state
                gradients = A.grad loss flatParameters
            when (iter `mod` 50 == 0) do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            newParam <- mapM A.makeIndependent
                $ sgd 1e-3 flatParameters gradients
            pure $ replaceParameters state newParam
    pure trained
    where
        spec = CNNSpec
        dataDim = 784
        numIters = 3000
        batchSize = 256


main :: IO ()
main = do
    let dataloader =
          Dataloader
            256     --- ^ batch size
            Nothing --- ^ shuffle
            cifar10 --- ^ datasets
            id      --- ^ transform
        stream = batchStream dataloader
      
    model <- train trainData

    -- show test images + labels
    mapM (\idx -> do
        testImg <- UI.getImages' 1 784 trainData [idx]
        UI.dispImage testImg
        putStrLn $ "Model        : " ++ (show . (argmax (Dim 1) RemoveDim) . exp $ cnn model testImg)
        putStrLn $ "Ground Truth : " ++ (show $ UI.getLabels' 1 trainData [idx])
        ) [0..10]

    putStrLn "Done"
