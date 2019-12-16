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

import qualified Image as I
import qualified UntypedImage as UI
import Common (foldLoop)

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

randomIndexes :: Int -> [Int]
randomIndexes size = (`mod` size) <$> randoms seed where seed = mkStdGen 123

mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} input = 
    logSoftmax 1
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input

train :: I.MnistData -> IO MLP
train trainData = do
    init <- sample spec
    let nImages = I.length trainData
        idxList = randomIndexes nImages
    trained <- foldLoop init numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            input <- UI.getImages' batchSize dataDim trainData idx
            let label = UI.getLabels' batchSize trainData idx
                loss = nllLoss' (mlp state input) label
                flatParameters = flattenParameters state
                gradients = A.grad loss flatParameters
            when (iter `mod` 50 == 0) do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            newParam <- mapM A.makeIndependent
                $ sgd 1e-3 flatParameters gradients
            pure $ replaceParameters state newParam
    pure trained
    where
        spec = MLPSpec 784 64 32 10
        dataDim = 784
        numIters = 3000
        batchSize = 256


main :: IO ()
main = do
    (trainData, testData) <- I.initMnist
    model <- train trainData

    -- show test images + labels
    mapM (\idx -> do
        testImg <- UI.getImages' 1 784 trainData [idx]
        UI.dispImage testImg
        putStrLn $ "Model        : " ++ (show . (argmax (Dim 1) RemoveDim) . exp $ mlp model testImg)
        putStrLn $ "Ground Truth : " ++ (show $ UI.getLabels' 1 trainData [idx])
        ) [0..10]

    putStrLn "Done"
