{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Control.Monad.Cont (runContT)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.State (evalStateT)
import Control.Monad.Trans (MonadTrans (..))
import qualified Data.List as List
import qualified Data.Set as Set
import Data.Word (Word64)
import GHC.Generics (Generic)
import qualified Hedgehog.Internal.Seed as Seed
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Prelude as P
import System.Random (mkStdGen)
import qualified Tokenizers
import Torch.GraduallyTyped
import qualified Dataset (STLCData (..), STLCExample (..))
import qualified Model (NeuralInterpreter (..))

-- | Data type for monitoring the training and evaluation losses.
data Monitor
  = -- | monitor for training loss
    TrainingMonitor {mtLoss :: Float, mtEpoch :: Int}
  | -- | monitor for evaluation loss
    EvaluationMonitor {meLoss :: Float, meEpoch :: Int}
  deriving stock (Eq, Ord, Show, Generic)

-- | A simple monitor that prints the training and evaluation losses to stdout.
monitor :: MonadIO m => P.Consumer Monitor m r
monitor = P.map show P.>-> P.stdoutLn'

main :: IO ()
main = Tokenizers.withTokenizerFromConfigFile "/tmp/t5-small-tokenizer.json" $ \tokenizer -> do
  let seed = 31415
      device = SDevice SCPU

  let -- during training, we need to turn dropout on and keep track of the gradient
      trainingModelSpec = Model.NeuralInterpreter $ t5SmallSpec SWithLMHead (SGradient SWithGradient) device SWithDropout
      -- during evaluation, we don't need to turn dropout on, nor do we need to keep track of the gradient
      evaluationModelSpec = Model.NeuralInterpreter $ t5SmallSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout

  -- initialize the model from the model specification
  stateDict <- stateDictFromFile "/tmp/t5-small-state-dict.pt"
  model <- flip evalStateT stateDict $ fromStateDict trainingModelSpec mempty

  let maxInputLength = 512
      maxTargetLength = 256

  -- buffered collation function that converts a stream of examples into one of batches
  let collate' =
        let maxBatchSize = 8
            collateFn chunk =
              let batchDim = SNoName :&: SUncheckedSize (fromIntegral $ length chunk)
                  chunk' = (\Dataset.STLCExample {..} -> (exInputIds, exTargetIds)) <$> chunk
                  (inputIds, targetIds) = unzip chunk'
                  maxInputLength' = min maxInputLength (foldr (max . length) 0 inputIds)
                  encoderSeqDim = SNoName :&: SUncheckedSize (fromIntegral maxInputLength')
                  maxTargetLength' = min maxTargetLength (foldr (max . length) 0 targetIds)
                  decoderSeqDim = SNoName :&: SUncheckedSize (fromIntegral maxTargetLength')
               in (,)
                    <$> mkT5Input batchDim encoderSeqDim device inputIds
                    <*> mkT5Input batchDim decoderSeqDim device targetIds
         in bufferedCollate (P.bounded 10) maxBatchSize collateFn

  let numEpochs = 100
      learningRateSchedule =
        let maxLearningRate = 1e-2
            finalLearningRate = 1e-4
            numWarmupEpochs = 10
            numCooldownEpochs = 10
         in singleCycleLearningRateSchedule maxLearningRate finalLearningRate numEpochs numWarmupEpochs numCooldownEpochs

  let -- create a dataset of unique training examples
      trainingLen = 10
      trainingData =
        Dataset.STLCData
          { name = "training",
            seeds =
              Set.fromList
                . List.take trainingLen
                $ Seed.from <$> List.iterate (+ 1) (0 :: Word64),
            maxInputLength,
            maxTargetLength,
            tokenizer
          }

      -- create a dataset of unique evaluation examples
      evaluationLen = 10
      evaluationData =
        Dataset.STLCData
          { name = "evaluation",
            seeds =
              Set.fromList
                . List.take evaluationLen
                $ Seed.from <$> List.iterate (+ 1) (fromInteger . toInteger $ trainingLen :: Word64),
            maxInputLength,
            maxTargetLength,
            tokenizer
          }

      -- configure the data loader with 1 worker thread and for random shuffling
      streamingState = (datasetOpts 1) {shuffle = Shuffle (mkStdGen 13)}

  -- create an Adam optimizer from the model
  optim <- liftIO $ mkAdam defaultAdamOptions {learningRate = 1e-4} model

  let -- one epoch of training and evaluation
      step (streamingState', g') epoch = do
        -- let learningRate = learningRateSchedule epoch
        -- setLearningRate optim learningRate

        -- helper function for streaming
        let go streamingState'' data' monitor' closure' = do
              (stream, shuffle) <- P.lift $ streamFromMap streamingState'' data'
              trainingLoss <- P.lift $ do
                batchedStream <- collate' stream
                lift $ closure' batchedStream
              case trainingLoss of
                Left g'' -> pure (g'', shuffle)
                Right (loss, g'') -> do
                  P.yield (monitor' (fromTensor loss) epoch)
                  pure (g'', shuffle)

        -- train for one epoch on the training set
        (g'', shuffle) <- go streamingState' trainingData TrainingMonitor $ \batchedStream ->
          train optim trainingModelSpec batchedStream g'

        -- evaluate on the evaluation set
        (g''', shuffle') <- go streamingState' {shuffle} evaluationData EvaluationMonitor $ \batchedStream -> do
          stateDict' <- getStateDict optim
          evaluationModel <- flip evalStateT stateDict' $ fromStateDict evaluationModelSpec mempty
          eval evaluationModel batchedStream g''

        pure (streamingState' {shuffle = shuffle'}, g''')

  -- create a Torch random generator from the seed
  g <- sMkGenerator device seed
  let -- initialize the training loop
      init' = pure (streamingState, g)

  let -- finalize the training loop
      done (_streamingState', _g) = pure ()

  -- run the training loop
  flip runContT pure . P.runEffect $
    P.foldM step init' done (P.each [1 .. numEpochs]) P.>-> monitor

  -- save the model's state dictionary to a file
  stateDict' <- getStateDict optim
  stateDictToFile stateDict' "neuralInterpreter.pt"
