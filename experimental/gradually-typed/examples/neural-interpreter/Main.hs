{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad (join)
import Control.Monad.Cont (runContT)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.State (evalStateT, runStateT, get)
import Control.Lens (Lens, Traversal, Lens', (^.), (%~))
import Control.Monad.Trans (MonadTrans (..))
import qualified Data.List as List
import qualified Data.Set as Set
import Data.Word (Word64)
import qualified Dataset (STLCData (..), STLCExample (..))
import GHC.Generics (Generic)
import qualified Hedgehog.Internal.Seed as Seed
import qualified Model (NeuralInterpreter (..))
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Prelude as P
import System.Random (mkStdGen)
import qualified Tokenizers
import qualified Control.Concurrent.MSem as MSem
import Torch.GraduallyTyped
import qualified Data.Text as Text

-- | Data type for monitoring the training and evaluation losses.
data Monitor
  = -- | monitor for training loss
    TrainingLossMonitor {mtLoss :: Float, mtEpoch :: Int}
  | -- | monitor for evaluation loss
    EvaluationLossMonitor {meLoss :: Float, meEpoch :: Int}
  | -- | monitor for predictions
    PredictionsMonitor {mpTargets :: [String], mpPredictions :: [String], mpEpoch :: Int}
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
  -- stateDict <- stateDictFromFile "/tmp/t5-small-state-dict.pt"
  stateDict <- stateDictFromFile "neuralInterpreter.pt"
  model <- flip evalStateT stateDict $ fromStateDict trainingModelSpec mempty

  let maxInputLength = 256
      maxTargetLength = 128

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
         in bufferedCollate (P.bounded 32) maxBatchSize collateFn

  let numEpochs = 100
      learningRateSchedule =
        let maxLearningRate = 1e-2
            finalLearningRate = 1e-4
            numWarmupEpochs = 10
            numCooldownEpochs = 10
         in singleCycleLearningRateSchedule maxLearningRate finalLearningRate numEpochs numWarmupEpochs numCooldownEpochs

  tokenizerSem <- MSem.new (1 :: Int)
  let tokenize input = MSem.with tokenizerSem $ do
        encoding <- Tokenizers.encode tokenizer input
        Tokenizers.getIDs encoding
      detokenize ids = MSem.with tokenizerSem $ do
        Tokenizers.decode tokenizer ids
    
  let -- create a dataset of unique training examples
      trainingLen = 256
      trainingData =
        Dataset.STLCData
          { name = "training",
            seeds =
              Set.fromList
                . List.take trainingLen
                $ Seed.from <$> List.iterate (+ 1) (0 :: Word64),
            maxInputLength,
            maxTargetLength,
            tokenize,
            detokenize
          }

      -- create a dataset of unique evaluation examples
      evaluationLen = 32
      evaluationData =
        Dataset.STLCData
          { name = "evaluation",
            seeds =
              Set.fromList
                . List.take evaluationLen
                $ Seed.from <$> List.iterate (+ 1) (fromInteger . toInteger $ trainingLen :: Word64),
            maxInputLength,
            maxTargetLength,
            tokenize,
            detokenize
          }

      -- configure the data loader with 1 worker thread and for random shuffling
      streamingState = (datasetOpts 2) {shuffle = Shuffle (mkStdGen 13)}

  -- create an Adam optimizer from the model
  optim <- liftIO $ mkAdam defaultAdamOptions {learningRate = 1e-4} model

  let -- one epoch of training and evaluation
      step (streamingState', g') epoch = do
        -- let learningRate = learningRateSchedule epoch
        -- setLearningRate optim learningRate

        -- helper function for streaming
        let go streamingState'' data' closure' = do
              (stream, shuffle) <- P.lift $ streamFromMap streamingState'' data'
              r <- P.lift $ do
                batchedStream <- collate' stream
                lift $ closure' batchedStream
              case r of
                Left g'' -> pure (g'', shuffle)
                Right (monitor', g'') -> do
                  P.yield (monitor' epoch)
                  pure (g'', shuffle)

        -- train for one epoch on the training set
        -- (g'', shuffle) <- go streamingState' trainingData $ \batchedStream -> do
        --   r <- train optim trainingModelSpec batchedStream g'
        --   pure $ (\(loss, g''') -> (TrainingLossMonitor (fromTensor loss), g''')) <$> r

        -- evaluate on the evaluation set
        (g''', _) <- go streamingState' {shuffle = Sequential} evaluationData $ \batchedStream -> do
          stateDict' <- getStateDict optim
          model' <- flip evalStateT stateDict' $ fromStateDict evaluationModelSpec mempty
          r <- eval model' batchedStream g'
          pure $ (\(loss, g'''') -> (EvaluationLossMonitor (fromTensor loss), g'''')) <$> r
        
        (g'''', _) <- go streamingState' {shuffle = Sequential} evaluationData $ \batchedStream -> do
          stateDict' <- getStateDict optim
          Model.NeuralInterpreter t5 <- flip evalStateT stateDict' $ fromStateDict evaluationModelSpec mempty
          x <- P.next (P.enumerate batchedStream)
          case x of
            Left _ -> pure . Left $ g'''
            Right ((encoderInput, decoderInput), producer) -> do
              let input = SimplifiedEncoderDecoderTransformerInput' encoderInput
                  [Dim _ batchSize, Dim _ _] = getDims encoderInput
                  batchDim = SNoName :&: SUncheckedSize batchSize
              (SimplifiedEncoderDecoderTransformerOutput' encoderOutput paddingMask, g'''') <- forward t5 input g'''
              x <- SimplifiedEncoderDecoderTransformerGenerationInput 
                    <$> mkTransformerInput
                          t5PadTokenId
                          batchDim
                          (SNoName :&: SUncheckedSize 0)
                          device
                          []
                    <*> pure encoderOutput
                    <*> pure paddingMask
              us <- sOnes $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device (SDataType SInt64) (SShape $ batchDim :|: SNil)
              ((SimplifiedEncoderDecoderTransformerGenerationInput decoderInput' _ _, g'''''), _us) <- flip runStateT us $ 
                decode (\input@(SimplifiedEncoderDecoderTransformerGenerationInput decoderInput' _ _) g -> do
                  let [Dim _ _, Dim _ seqLen] = getDims decoderInput'
                  unfinishedSequences <- get
                  b <- allSequencesFinished unfinishedSequences
                  if (b || seqLen >= fromIntegral maxTargetLength) then
                    pure Nothing
                  else do
                      (output, g') <- forward t5 input g
                      input' <- (sedtOutputToInput . prepNext %~ greedyNextTokens t5PadTokenId t5EOSTokenId) output
                      pure $ Just (input', g')
                ) x g''''

              let postProcess = Text.unpack . 
                    Text.replace "<unk>" "\\" .
                    Text.replace "<pad>" mempty . Text.pack
              let decoderIds :: [[Int]] = fromTensor decoderInput
              targets <- traverse (Tokenizers.decode tokenizer) decoderIds
              let targets' = postProcess <$> targets
              let decoderIds' :: [[Int]] = fromTensor decoderInput'
              predictions <- traverse (Tokenizers.decode tokenizer) decoderIds'
              let predictions' = postProcess <$> predictions

              pure . Right $ (PredictionsMonitor targets' predictions', g''''')

        pure (streamingState', g'''')

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
  -- stateDict' <- getStateDict optim
  -- stateDictToFile stateDict' "neuralInterpreter.pt"
