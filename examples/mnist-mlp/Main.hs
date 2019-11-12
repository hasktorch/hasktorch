{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import GHC.Generics

import Torch.Autograd as A
import Torch.Functions
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

mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} input = 
    linear' l2
    . relu
    . linear' l1
    . relu
    . linear' l0
    $ input

train trainingData = do
    init <- sample spec
    trained <- foldLoop init numIters $ 
        \state -> do
            let input = undefined -- TODO - setup minibatching
            let prediction = mlp state input
            let flatParameters = flattenParameters state
            let loss = undefined -- TODO port loss calculation
            let gradients = A.grad loss flatParameters
            pure undefined -- TODO - iteration loop
    pure ()
    where
    spec = MLPSpec 768 10 512 256 
    numIters = 1000000

main :: IO ()
main = do
    (trainData, testData) <- I.initMnist
    let labels = UI.getLabels' 10 trainData [0..100]
    let images = UI.getImages
    print labels
    putStrLn "Done"
