{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import qualified Codec.Picture as I
import Control.Monad (when)
import GHC.Generics
import Numeric.Dataloader
import Numeric.Datasets
import Numeric.Datasets.CIFAR10
import qualified Streaming as S
import qualified Streaming.Prelude as S
import System.Random (mkStdGen, randoms)
import Torch
import Torch.Vision
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
  deriving (Generic, Show)

instance Parameterized MLP

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} =
    MLP
      <$> sample (LinearSpec inputFeatures hiddenFeatures0)
      <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
      <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} =
  logSoftmax (Dim 1)
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0

foldLoop' :: a -> S.Stream (S.Of [b]) IO () -> (a -> [b] -> IO a) -> IO a
foldLoop' x dat block = S.foldM_ block (pure x) pure dat

train :: Int -> S.Stream (S.Of [CIFARImage]) IO () -> IO MLP
train numEpoch trainData = do
  init <- sample spec :: IO MLP
  foldLoop init numEpoch $ \state0 iter -> do
    (trained', trained_loss') <- foldLoop' (state0, 0) trainData $ \(state, sumLoss) batch -> do
      images <- fromImages $ map (fst . getXY) batch
      let len = length batch
          input = toType Float $ reshape [len, 1024 * 3] images
          label = asTensor $ map (fromEnum . snd . getXY) batch
          loss = nllLoss' label $ mlp state input
          flatParameters = flattenParameters state
      (newParam, _) <- runStep state GD loss 1e-3
      pure (newParam, toDouble loss + sumLoss)
    putStrLn $ "Epoch: " ++ show iter ++ " | Loss: " ++ show trained_loss'
    pure trained'
  where
    spec = MLPSpec (1024 * 3) 64 32 10

main :: IO ()
main = do
  let dataloader =
        Dataloader
          256 --- ^ batch size
          Nothing --- ^ shuffle
          cifar10 --- ^ datasets
          id --- ^ transform
      stream = batchStream dataloader
      numEpoch = 3
  model <- train numEpoch stream

  putStrLn "Done"
