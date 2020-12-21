{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import Control.Exception.Safe
  ( SomeException (..),
    try,
  )
import Control.Monad (forM_, when, (<=<))
import Control.Monad.Cont (ContT (..))
import GHC.Generics
import Pipes
import qualified Pipes.Prelude as P
import System.Environment
import Torch
import Torch.Serialize
import Torch.Typed.Vision (initMnist)
import qualified Torch.Vision as V
import Prelude hiding (exp)

data MLPSpec = MLPSpec
  { inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
  }
  deriving (Show, Eq)

data MLP = MLP
  { l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
  }
  deriving (Generic, Show, Parameterized, ToTensor)

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} =
    MLP
      <$> sample (LinearSpec inputFeatures hiddenFeatures0)
      <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
      <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input =
  logSoftmax (Dim 1)
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input

trainLoop :: Optimizer o => (Device, DType) -> MLP -> o -> ListT IO (Tensor, Tensor) -> IO MLP
trainLoop device' model optimizer = P.foldM step begin done . enumerateData
  where
    step :: MLP -> ((Tensor, Tensor), Int) -> IO MLP
    step model args = do
      let ((input, label), iter) = toDevice' device' args
          loss = nllLoss' label $ mlp model input
      when (iter `mod` 50 == 0) $ do
        putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
      (newParam, _) <- runStep model optimizer loss 1e-3
      pure newParam
    done = pure
    begin = pure model

displayImages :: MLP -> (Tensor, Tensor) -> IO ()
displayImages model (testImg, testLabel) = do
  V.dispImage testImg
  putStrLn $ "Model        : " ++ (show . (argmax (Dim 1) RemoveDim) . exp $ mlp model testImg)
  putStrLn $ "Ground Truth : " ++ (show testLabel)

toDevice' :: ToTensor a => (Device, DType) -> a -> a
toDevice' (device', dtype') model = gmap func model
  where
    func tensor =
      let tensor' = toDevice device' tensor
          tensor'' =
            if isIntegral (dtype tensor')
            then tensor'
            else toType dtype' tensor'
      in tensor''

fromDevice :: ToTensor a => a -> a
fromDevice model = gmap func model
  where
    func tensor =
      let tensor' = toDevice (Device CPU 0) tensor
          tensor'' =
            if isIntegral (dtype tensor')
            then tensor'
            else toType Float tensor'
      in tensor''


main :: IO ()
main = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  dtypeStr <- try (getEnv "DTYPE") :: IO (Either SomeException String)
  let localDevice = case deviceStr of
        Right "cpu" -> Device CPU 0
        Right "cuda:0" -> Device CUDA 0
        Right device -> error $ "Unknown device setting: " ++ device
        _ -> Device CPU 0
      localDType = case dtypeStr of
        Right "double" -> Double
        Right "float" -> Float
        Right "half" -> Half
        Right "bfloat16" -> BFloat16
        Right type' -> error $ "Unknown dtype setting: " ++ type'
        _ -> Float
  (trainData, testData) <- initMnist "data"
  let trainMnist = V.MNIST {batchSize = 32, mnistData = trainData}
      testMnist = V.MNIST {batchSize = 1, mnistData = testData}
      spec = MLPSpec 784 64 32 10
      optimizer = GD
      device' = (localDevice, localDType)
  init <- toDevice' device' <$> sample spec
  model <- foldLoop init 5 $ \model _ ->
    runContT (streamFromMap (datasetOpts 2) trainMnist) $ trainLoop device' model optimizer . fst

  -- show test images + labels
  forM_ [0 .. 10] $ displayImages (fromDevice model) <=< getItem testMnist

  putStrLn "Done"
