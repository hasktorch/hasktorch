{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}

module Monitor where

import Control.Monad.IO.Class (MonadIO (liftIO))
import Data.Aeson (ToJSON (toEncoding))
import Data.Aeson.Encoding (encodingToLazyByteString)
import Data.Aeson.TH (defaultOptions, deriveJSON)
import qualified Data.ByteString as BS
import Data.ByteString.Lazy (snoc, toStrict)
import Data.Foldable (toList)
import qualified Data.List as List
import qualified Data.Text.Lazy as Text.Lazy
import qualified Dataset (STLCExample (..))
import GHC.Generics (Generic)
import qualified Opts
import Pipes ((>->))
import qualified Pipes as P
import qualified Pipes.Prelude as P hiding (toHandle)
import qualified System.IO as IO
import qualified System.ProgressBar as PB

-- | Data type for monitoring the training and evaluation metrics.
data Monitor
  = -- | monitor for configuration
    ConfigMonitor
      {mcConfig :: !Opts.Config}
  | -- | monitor for training metrics
    TrainingMonitor
      { mtLoss :: !Float,
        mtEpoch :: !Int
      }
  | -- | monitor for evaluation metrics
    EvaluationMonitor
      { meLoss :: !Float,
        meTargets :: ![String],
        mePredictions :: ![String],
        meExactMatchAccuracy :: !Float,
        meExamples :: ![Dataset.STLCExample Int],
        meEpoch :: !Int
      }
  deriving stock (Eq, Ord, Show, Generic)

$(deriveJSON defaultOptions ''Monitor)

-- | A simple monitor that prints the training and evaluation metrics to stdout
-- and saves the predictions to a file handle.
monitor :: MonadIO m => Int -> IO.Handle -> P.Consumer Monitor m r
monitor numEpochs h =
  P.tee showProgress
    >-> P.map (toStrict . flip snoc 0x0a . encodingToLazyByteString . toEncoding)
    >-> P.for P.cat (\bs -> liftIO $ BS.hPut h bs >> IO.hFlush h)
  where
    showProgress = do
      let maxRefreshRate = 10
          epoch =
            let render progress timing = "epoch " <> PB.runLabel PB.exact progress timing
             in PB.Label render
      let metrics =
            let render PB.Progress {..} _ =
                  let (trainingLoss, evaluationLoss, exactMatchAccuracy) = progressCustom
                   in Text.Lazy.pack $
                        List.intercalate
                          " | "
                          ( List.concat
                              [ toList $ (\x -> "training loss: " <> show x) <$> trainingLoss,
                                toList $ (\x -> "evaluation loss: " <> show x) <$> evaluationLoss,
                                toList $ (\x -> "exact match accuracy: " <> show x) <$> exactMatchAccuracy
                              ]
                          )
             in PB.Label render
      pb <-
        P.lift . liftIO $
          PB.newProgressBar
            PB.defStyle
              { PB.stylePrefix = epoch,
                PB.stylePostfix = metrics
              }
            maxRefreshRate
            (PB.Progress 0 numEpochs (Nothing, Nothing, Nothing))
      P.for P.cat $ \case
        ConfigMonitor {} -> pure ()
        TrainingMonitor {..} ->
          P.lift . liftIO $
            PB.updateProgress
              pb
              ( \progress ->
                  let (_trainingLoss, evaluationLoss, exactMatchAccuracy) = PB.progressCustom progress
                   in progress {PB.progressDone = mtEpoch, PB.progressCustom = (Just mtLoss, evaluationLoss, exactMatchAccuracy)}
              )
        EvaluationMonitor {..} ->
          P.lift . liftIO $
            PB.updateProgress
              pb
              ( \progress ->
                  let (trainingLoss, _evaluationLoss, _exactMatchAccuracy) = PB.progressCustom progress
                   in progress {PB.progressDone = meEpoch, PB.progressCustom = (trainingLoss, Just meLoss, Just meExactMatchAccuracy)}
              )