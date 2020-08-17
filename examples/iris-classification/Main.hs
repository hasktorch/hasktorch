{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
module Main where

import           Control.Arrow (Arrow(first))
import           Control.Monad (foldM, when, (>=>))
import           Control.Monad.Cont (runContT, runCont, ContT(ContT))
import           Data.Csv (FromNamedRecord)
import           Data.Vector (Vector, toList)
import           GHC.Generics (Generic)
import           Pipes
import qualified Pipes.Prelude as P
import           Pipes.Safe (runSafeT)
import           Torch
import           Torch.Data.CsvDataset
import           Torch.Data.StreamedPipeline 
import           Torch.Tensor

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
    "Iris-setosa"      -> pure Setosa
    "Iris-versicolor" -> pure Versicolor
    "Iris-virginica"   -> pure Virginica
    _                  -> mzero

data Iris = Iris { sepalLength :: Float
                 , sepalWidth  :: Float
                 , petalLength :: Float
                 , petalWidth  :: Float
                 , irisClass   :: IrisClass
                 } deriving (Generic, Show, FromRecord)

irisToTensor :: Vector Iris -> (Tensor, Tensor)
irisToTensor iris = do
  -- want to only traverse this list once 
  (asTensor . toList $ getFeatures iris, toType Float $ oneHot 3 (asTensor . toList $ getClasses iris) )
  where 
        getFeatures = fmap (\x -> [sepalLength x, sepalWidth x, petalLength x, petalWidth x])
        getClasses = fmap (\x -> fromEnum $ irisClass x)

trainLoop :: (Optimizer o, MonadIO m) => MLP -> o -> ListT m ((Tensor, Tensor), Int)  -> m MLP
trainLoop model optimizer inputs = P.foldM step init done $ enumerate inputs
  where 
        step model ((input, label), iter) = do
          let loss = binaryCrossEntropyLoss' label $ mlp model input
          when ((iter `mod` 5) == 0) $ do
            liftIO $ print iter
            liftIO $ putStrLn $ " | Loss: " ++ show loss
          (newParam, _) <- liftIO $ runStep model optimizer loss 1e-2
          pure $ replaceParameters model newParam

        done = pure
        init = pure model

main :: IO ()
main = runSafeT $ do
  init <- liftIO $ sample spec 
  let (irisTrain :: CsvDataset Iris) = (csvDataset  "data/iris.data") { batchSize = 1
                                                                      -- need to bring whole dataset into memory to get a good shuffle
                                                                      -- since iris.data is sorted
                                                                      , shuffle = Just 150 
                                                                      }
  foldM (\model epoch -> do
            flip runContT (trainLoop model optimizer) $ do raw <- makeListT' dataloaderOpts irisTrain [()]
                                                           pmap 1 (first irisToTensor) raw
        ) init [1..10]

  pure ()
  where spec = MLPSpec 4 10 10 3
        optimizer = GD
  
