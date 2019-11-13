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
import Common (foldLoop, randomIndexes)

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


train trainData = do
    init <- sample spec
    let nImages = I.length trainData
    let idx = randomIndexes nImages
    randomized <- UI.getImages' nImages dataDim trainData idx
    trained <- foldLoop init numIters $ 
        \state -> do
            let input = undefined -- TODO - setup minibatching
            let label = undefined -- TODO - setup minibatching
            let prediction = mlp state input
            let flatParameters = flattenParameters state
            let loss = binary_cross_entropy_loss' prediction label
            let gradients = A.grad loss flatParameters
            pure undefined -- TODO - iteration loop
    pure ()
    where
    spec = MLPSpec 768 10 512 256 
    dataDim = 768
    numIters = 1000000
    batchSize = 512

main :: IO ()
main = do
    (trainData, testData) <- I.initMnist
    let labels = UI.getLabels' 10 trainData [0..100]
    print labels
    -- train trainData
    putStrLn "Done"
