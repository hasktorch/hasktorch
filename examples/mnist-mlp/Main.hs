{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import           Control.Monad (when)
import           GHC.Generics
import           Prelude hiding (exp)

import           Torch
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V
import           Torch.Serialize

import           Control.Monad (forever)
import           Control.Monad.Cont (ContT(runContT))
import           Pipes
import qualified Pipes.Prelude as P
import           Torch.Data.StreamedPipeline
import           Torch.Data.Pipeline
import Control.Monad (forM_)

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

displayImages :: MLP -> (Tensor, Tensor) -> IO ()
displayImages model (testImg, testLabel) =  do
  -- ((testImg, testLabel), _) <- await
  V.dispImage testImg
  putStrLn $ "Model        : " ++ (show . (argmax (Dim 1) RemoveDim) . exp $ mlp model testImg)
  putStrLn $ "Ground Truth : " ++ (show $ testLabel)

main :: IO ()
main = do
    (trainData, testData) <- V.initMnist "data"
    let trainMnist = V.Mnist { batchSize = 64 , mnistData = trainData}
        testMnist = V.Mnist { batchSize = 1 , mnistData = testData}
        spec = MLPSpec 784 64 32 10
        optimizer = GD
    init <- sample spec
    model <- foldLoop init 5 $ \model _ ->
      runContT (readBatches (mapStyleOpts 1) trainMnist sequentialSampler Torch.Data.StreamedPipeline.cat) (trainLoop model optimizer)
  
    -- show test images + labels
    forM_ [0..10]  (displayImages model . getItem testMnist)

    putStrLn "Done"
