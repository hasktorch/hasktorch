{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad (foldM, when)
import Control.Monad.Cont (runContT)
import Data.Vector (Vector, toList)
import GHC.Generics (Generic)
import Pipes
import Pipes.Concurrent (unbounded)
import qualified Pipes.Prelude as P
import Pipes.Safe (runSafeT)
import Torch
import Torch.Data.CsvDatastream

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
  deriving (Generic, Show)

instance Parameterized MLP

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} =
    MLP
      <$> sample (LinearSpec inputFeatures hiddenFeatures0)
      <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
      <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input =
  softmax (Dim 1)
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input

data IrisClass = Setosa | Versicolor | Virginica deriving (Eq, Show, Enum, Generic)

instance FromField IrisClass where
  parseField b = case b of
    "Iris-setosa" -> pure Setosa
    "Iris-versicolor" -> pure Versicolor
    "Iris-virginica" -> pure Virginica
    _ -> mzero

data Iris = Iris
  { sepalLength :: Float,
    sepalWidth :: Float,
    petalLength :: Float,
    petalWidth :: Float,
    irisClass :: IrisClass
  }
  deriving (Generic, Show, FromRecord)

irisToTensor :: Vector Iris -> (Tensor, Tensor)
irisToTensor iris = do
  -- want to only traverse this list once
  (asTensor . toList $ getFeatures iris, toType Float $ oneHot 3 (asTensor . toList $ getClasses iris))
  where
    getFeatures = fmap (\x -> [sepalLength x, sepalWidth x, petalLength x, petalWidth x])
    getClasses = fmap (\x -> fromEnum $ irisClass x)

trainLoop :: (Optimizer o, MonadIO m) => MLP -> o -> ListT m (Tensor, Tensor) -> m MLP
trainLoop model optimizer inputs = P.foldM step init done $ enumerateData inputs
  where
    step model ((input, label), iter) = do
      let loss = binaryCrossEntropyLoss' label $ mlp model input
      when ((iter `mod` 5) == 0) $ do
        liftIO $ print iter
        liftIO $ putStrLn $ " | Loss: " ++ show loss
      (newParam, _) <- liftIO $ runStep model optimizer loss 1e-2
      pure newParam

    done = pure
    init = pure model

main :: IO ()
main = runSafeT $ do
  init <- liftIO $ sample spec
  let (irisTrain :: CsvDatastream Iris) =
        (csvDatastream "data/iris.data")
          { batchSize = 1,
            -- need to bring whole dataset into memory to get a good shuffle
            -- since iris.data is sorted
            bufferedShuffle = Just 150
          }
  foldM
    ( \model epoch ->
        runContT (streamFrom' datastreamOpts irisTrain [()] >>= pmap unbounded irisToTensor) $
          trainLoop model optimizer
    )
    init
    [1 .. 10]

  pure ()
  where
    spec = MLPSpec 4 10 10 3
    optimizer = GD
