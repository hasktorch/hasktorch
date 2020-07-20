{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
module Main where

import           Control.Monad (foldM, when, (>=>))
import           Data.Csv (FromNamedRecord)
import           GHC.Generics (Generic)
import           Pipes
import qualified Pipes.Prelude as P
import           Pipes.Safe (runSafeT)
import           Torch
import           Torch.Tensor
import           Torch.Data.CsvDataset
import           Torch.Data.Pipeline (FoldM(FoldM))
import           Torch.Data.StreamedPipeline (pmap, makeListT)
import Data.Vector (Vector, toList)
import Control.Arrow (Arrow(first))


data MLPSpec = MLPSpec {
    inputFeatures   :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures  :: Int
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

data IrisClass = Setosa | Versicolor | Virginica deriving (Eq, Show, Enum, Generic)

instance FromField IrisClass where
  parseField b = case b of
    "Iris-setosa"      -> pure Setosa
    "Iris-versicolor" -> pure Versicolor
    "Iris-virginica"   -> pure Virginica
    _                  -> mzero

data Iris = Iris { sepalLength :: Float
                 , sepalWidth  :: Float
                 , petalLength :: Float
                 , petalWidth  :: Float
                 , irisClass   :: IrisClass
                 } deriving (Generic, Show, FromRecord, FromNamedRecord)

irisToTensor :: Vector Iris -> (Tensor, Tensor)
irisToTensor iris = do
  -- want to only traverse this list once 
  (asTensor . toList $ getFeatures iris, asTensor . toList $ getClasses iris)
  where 
        getFeatures = fmap (\x -> [sepalLength x, sepalWidth x, petalLength x, petalWidth x])
        getClasses = fmap (\x -> fromEnum $ irisClass x)

trainLoop :: (Optimizer o, MonadIO m) => MLP -> o -> Producer ((Tensor, Tensor), Int) m () -> m MLP
trainLoop model optimizer = P.foldM step init done
  where 
        step model ((input, label), iter) = do
          let loss = nllLoss' label $ mlp model input
          when (iter `mod` 100 == 0) $ do
            liftIO $ putStrLn $ " | Loss: " ++ show loss
            -- liftIO $ print label
            -- liftIO $ print $ mlp model input
          (newParam, _) <- liftIO $ runStep model optimizer loss 1e-3
          pure $ replaceParameters model newParam
        done = pure
        init = pure model

pipeline dataset = makeListT @(Vector Iris) dataset >=> pmap (first irisToTensor ) 1

main :: IO ()
main = runSafeT $ do
  init <- liftIO $ sample spec 
  let irisTrain = (csvDataset @Iris "iris.data") { batchSize = 4 , shuffle = Just 500}
  foldM (\model epoch -> do
            (inputs :: ListT m ((Tensor, Tensor), Int)) <- pipeline irisTrain (Select $ yield ())
            trainLoop model optimizer $ enumerate inputs
            

        ) init [1..50]

  pure ()
  where spec = MLPSpec 4 100 100 3
        optimizer = GD
  
