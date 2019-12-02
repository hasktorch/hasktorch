{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (when)
import GHC.Generics
import System.Random (mkStdGen, randoms)

import Torch.Autograd as A
import Torch.Functions hiding (take)
import Torch.Tensor
import Torch.NN

import qualified Image as I
import qualified UntypedImage as UI
import Common (foldLoop)

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    outputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int
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
    . linear' l2
    . relu
    . linear' l1
    . relu
    . linear' l0
    $ input

train :: I.MnistData -> IO MLP
train trainData = do
    init <- sample spec
    let nImages = I.length trainData
        idxList = randomIndexes nImages
    trained <- foldLoop init numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            -- print (head idx)
            input <- UI.getImages' batchSize dataDim trainData idx
            let label = UI.getLabels' batchSize trainData [0..50000] -- TODO - review/fix chunking logic
                loss = nllLoss' (mlp state input) label
                flatParameters = flattenParameters state
                gradients = A.grad loss flatParameters
            when (iter `mod` 20 == 0) do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            newParam <- mapM A.makeIndependent
                $ sgd 1e-04 flatParameters gradients
            pure $ replaceParameters state newParam
    pure trained
    where
        spec = MLPSpec 784 10 64 32
        dataDim = 784
        numIters = 1000
        batchSize = 256


main :: IO ()
main = do
    (trainData, testData) <- I.initMnist

    -- print a few labels 
    let labels = UI.getLabels' 10 trainData [0..100]
    print labels

    train trainData

    -- show test images + labels
    foldLoop undefined 5 \_ idx -> do
        testImg <- reshape [28, 28] <$> UI.getImages' 1 784 trainData [idx]
        UI.dispImage testImg
        putStrLn $ "Label : " ++ (show $ UI.getLabels' 1 trainData [idx])

    putStrLn "Done"
