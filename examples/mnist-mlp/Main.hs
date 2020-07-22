{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (when)
import GHC.Generics
import System.Random (mkStdGen, randoms)
import Prelude hiding (exp)

import Torch
import Pipes
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V
import Torch.Serialize

import Torch.Data.StreamedPipeline
import qualified Pipes.Prelude as P
import Torch.Data.Pipeline (FoldM(FoldM))
import Control.Monad.Cont (ContT(runContT))

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
    logSoftmax (Dim 1)
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input

trainLoop :: Optimizer o => MLP -> o -> ListT IO ((Tensor, Tensor), Int) -> IO  MLP
trainLoop model optimizer = P.foldM  step begin done . enumerate
  where step :: MLP -> ((Tensor, Tensor), Int) -> IO MLP
        step model ((input, label), iter) = do
          let loss = nllLoss' label $ mlp model input
          when (iter `mod` 50 == 0) $ do
            putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
          (newParam, _) <- runStep model optimizer loss 1e-3
          pure $ replaceParameters model newParam
        done = pure
        begin = pure model

main :: IO ()
main = do
    (trainData, testData) <- V.initMnist "data"
    let trainMnist = V.Mnist { batchSize = 256 , mnistData = trainData}
        testMnist = V.Mnist { batchSize = 256 , mnistData = testData}
        spec = MLPSpec 784 64 32 10
        optimizer = GD
    init <- sample spec
    model <- flip runContT (trainLoop init optimizer) $ makeListT' trainMnist [1 :: Int]

    -- show test images + labels
    mapM (\idx -> do
        testImg <- V.getImages' 1 784 testData [idx]
        V.dispImage testImg
        putStrLn $ "Model        : " ++ (show . (argmax (Dim 1) RemoveDim) . exp $ mlp model testImg)
        putStrLn $ "Ground Truth : " ++ (show $ V.getLabels' 1 testData [idx])
        ) [0..10]

    putStrLn "Done"
