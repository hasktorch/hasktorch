{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import GHC.Generics

import Torch.Autograd as A
import Torch.Functions hiding (take)
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

train :: I.MnistData -> IO ()
train trainData = do
    init <- sample spec
    let nImages = I.length trainData
        idxList = randomIndexes nImages
    trained <- foldLoop init numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            input <- UI.getImages' nImages dataDim trainData idx
            let label = undefined -- TODO
            let prediction = mlp state input
                flatParameters = flattenParameters state
                loss = binary_cross_entropy_loss' prediction label
                gradients = A.grad loss flatParameters
            newParam <- mapM A.makeIndependent
                $ sgd 1e-01 flatParameters gradients
            pure $ replaceParameters state newParam
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
