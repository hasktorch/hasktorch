{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
module Main where

import           Control.Monad (foldM, when)
import           Data.Csv (FromNamedRecord)
import           GHC.Generics (Generic)
import           Pipes
import qualified Pipes.Prelude as P
import           Pipes.Safe (runSafeT)
import           Torch
import           Torch.Tensor
import           Torch.Data.CsvDataset
import           Torch.Data.Pipeline (FoldM(FoldM))
import           Torch.Data.StreamedPipeline (pMap, defaultDataloaderOpts, makeListT)



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

-- without this explicit type signature it can't find the right overlapping instance!
toTensors :: [Iris] -> (Tensor, Tensor)
toTensors iris = do
  -- Iris{..} <- iris 
  -- let features = [sepalLength , sepalWidth , petalLength , petalWidth]
  --     classes = fromEnum irisClass 
  -- pure (asTensor features, asTensor classes)

  -- yucky
  (asTensor $ getFeatures iris, asTensor $ getClasses iris)
  where 
        getFeatures = fmap (\x -> [sepalLength x, sepalWidth x, petalLength x, petalWidth x])
        getClasses = fmap (\x -> fromEnum $ irisClass x)

-- trainLoop :: Optimizer o => model -> o -> Producer [Iris] m  () -> m model
trainLoop model optimizer = P.foldM step init done
  -- where step :: MonadIO m => MLP -> (Tensor, Tensor) -> m MLP
  where 
        step model (input, label ) = do
        -- step model bad = do
          -- let (input, label) = Main.toTensors bad
          let loss = nllLoss' label $ mlp model input
          -- when (iter `mod` 50 == 0) $ do
          liftIO $ putStrLn $ " | Loss: " ++ show loss
          liftIO $ print label 
          (newParam, _) <- liftIO $ runStep model optimizer loss 1e-3
          pure $ replaceParameters model newParam
        done = pure
        init = pure model

main :: IO ()
main = runSafeT $ do
  init <- liftIO $ sample spec 
  let irisTrain = (csvDataset @Iris "iris.data") { batchSize = 4 , shuffle = Just 500}

  -- inputs <- makeListT @_ @_ @_ @[Iris] defaultDataloaderOpts irisTrain (Select $ yield ()) 
  -- transformed <- pMap inputs Main.toTensors 2

  foldM (\model epoch -> do
            inputs <- makeListT @_ @_ @_ @[Iris] defaultDataloaderOpts irisTrain (Select $ yield ()) 
            transformed <- pMap inputs Main.toTensors 2
            -- liftIO $ print epoch
            trainLoop model optimizer $ enumerate transformed
        ) init [1]
  -- liftIO $ foldM init 1000 $ trainLoop init (optimizer ) $ enumerate transformed 
  -- trainLoop init (optimizer init) $ enumerate inputs
  pure ()
  -- model <- foldOverWith' irisTrain (Select $ yield ()) (trainLoop (pure init) optimizer)
  where spec = MLPSpec 4 100 100 3
        optimizer = GD
  
