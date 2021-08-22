{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Control.Applicative ((<**>))
import qualified Control.Concurrent.MSem as MSem
import Control.Lens ((%~))
import Control.Monad.Cont (runContT)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.State (evalStateT, get, runStateT)
import Control.Monad.Trans (MonadTrans (..))
import Data.Int (Int16)
import qualified Data.List as List
import qualified Data.Set as Set
import qualified Data.Text as Text
import Data.Word (Word64)
import qualified Dataset (STLCData (..), STLCExample (..))
import GHC.Generics (Generic)
import qualified Hedgehog.Internal.Seed as Seed
import qualified Model (NeuralInterpreter (..))
import qualified Options.Applicative as Opts
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Prelude as P
import System.Random (mkStdGen)
import qualified Tokenizers
import Torch.GraduallyTyped

data ModelArchitecture = T5Small | T5Large | T5ThreeB | BARTBase | BARTLarge
  deriving stock (Show, Generic)

data Config = Config
  { configNumTrainingExamples :: Int,
    configNumEvaluationExamples :: Int,
    configNumEpochs :: Int,
    configNumWorkers :: Int,
    configBufferSize :: Int,
    configDevice :: DeviceType Int16,
    configModelArchitecture :: ModelArchitecture,
    configModelPretrainedPath :: Maybe FilePath,
    configModelSavePath :: FilePath,
    configTokenizerPath :: FilePath,
    configSeed :: Word64,
    configMaxInputLength :: Int,
    configMaxTargetLength :: Int,
    configMaxBatchSize :: Int,
    configLearningRate :: Double
  }
  deriving stock (Show, Generic)

config :: Opts.Parser Config
config =
  Config
    <$> Opts.option
      Opts.auto
      (Opts.long "num-training-examples" <> Opts.short 'n' <> Opts.value 65536 <> Opts.showDefault <> Opts.help "Number of training examples")
    <*> Opts.option
      Opts.auto
      (Opts.long "num-evaluation-examples" <> Opts.short 'e' <> Opts.value 4096 <> Opts.showDefault <> Opts.help "Number of evaluation examples")
    <*> Opts.option
      Opts.auto
      (Opts.long "num-epochs" <> Opts.short 'e' <> Opts.value 10 <> Opts.showDefault <> Opts.help "Number of epochs")
    <*> Opts.option
      Opts.auto
      (Opts.long "num-workers" <> Opts.short 'w' <> Opts.value 4 <> Opts.showDefault <> Opts.help "Number of workers for data loading")
    <*> Opts.option
      Opts.auto
      (Opts.long "buffer-size" <> Opts.short 'b' <> Opts.value 100 <> Opts.showDefault <> Opts.help "Buffer size for streaming")
    <*> Opts.option
      ( Opts.auto >>= \case
          -1 -> pure CPU
          i -> pure (CUDA i)
      )
      (Opts.long "device" <> Opts.short 'd' <> Opts.value CPU <> Opts.showDefault <> Opts.help "Device. -1 for CPU, 0 for CUDA 0, 1 for CUDA 1, etc.")
    <*> Opts.option
      ( Opts.auto >>= \case
          (0 :: Int) -> pure T5Small
          1 -> pure T5Large
          2 -> pure T5ThreeB
          3 -> pure BARTBase
          4 -> pure BARTLarge
          _ -> Opts.readerError "Invalid model architecture"
      )
      (Opts.long "model-architecture" <> Opts.short 'a' <> Opts.value T5Small <> Opts.showDefault <> Opts.help "Model architecture")
    <*> Opts.optional
      (Opts.strOption (Opts.long "model-pretrained-path" <> Opts.short 'p' <> Opts.help "Path to pretrained model. If not specified, a new model will be trained"))
    <*> Opts.strOption
      (Opts.long "model-save-path" <> Opts.short 's' <> Opts.help "Path to save model")
    <*> Opts.strOption
      (Opts.long "tokenizer-path" <> Opts.short 't' <> Opts.help "Path to pretrained tokenizer.")
    <*> Opts.option
      Opts.auto
      (Opts.long "seed" <> Opts.short 's' <> Opts.value 31415 <> Opts.showDefault <> Opts.help "Seed")
    <*> Opts.option
      Opts.auto
      (Opts.long "max-input-length" <> Opts.short 'i' <> Opts.value 256 <> Opts.showDefault <> Opts.help "Maximum input length in tokens")
    <*> Opts.option
      Opts.auto
      (Opts.long "max-target-length" <> Opts.short 't' <> Opts.value 128 <> Opts.showDefault <> Opts.help "Maximum target length in tokens")
    <*> Opts.option
      Opts.auto
      (Opts.long "max-batch-size" <> Opts.short 'b' <> Opts.value 24 <> Opts.showDefault <> Opts.help "Maximum batch size")
    <*> Opts.option
      Opts.auto
      (Opts.long "learning-rate" <> Opts.short 'l' <> Opts.value 0.0001 <> Opts.showDefault <> Opts.help "Learning rate")

opts :: Opts.ParserInfo Config
opts = Opts.info (config <**> Opts.helper) (Opts.fullDesc <> Opts.progDesc "Train a neural interpreter model on a synthetic dataset")

-- | Data type for monitoring the training and evaluation losses.
data Monitor
  = -- | monitor for training loss
    TrainingLossMonitor {mtLoss :: Float, mtEpoch :: Int}
  | -- | monitor for evaluation loss
    EvaluationLossMonitor {meLoss :: Float, meEpoch :: Int}
  | -- | monitor for predictions
    PredictionsMonitor {mpTargets :: [String], mpPredictions :: [String], mpExactMatchAccuracy :: Float, mpEpoch :: Int}
  deriving stock (Eq, Ord, Show, Generic)

-- | A simple monitor that prints the training and evaluation losses to stdout.
monitor :: MonadIO m => P.Consumer Monitor m r
monitor = P.map show P.>-> P.stdoutLn'

go Config {..} device trainingModelSpec evaluationModelSpec =
  Tokenizers.withTokenizerFromConfigFile configTokenizerPath $ \tokenizer -> do
    -- create a Torch random generator from the seed
    g0 <- sMkGenerator device configSeed

    -- initialize the model from the model specification
    (model, g1) <- case configModelPretrainedPath of
      Nothing -> initialize trainingModelSpec g0
      Just path -> do
        stateDict <- stateDictFromFile path
        model <- flip evalStateT stateDict $ fromStateDict trainingModelSpec mempty
        pure (model, g0)

    -- buffered collation function that converts a stream of examples into one of batches
    let collate' =
          let collateFn chunk =
                let batchDim = SNoName :&: SUncheckedSize (fromIntegral $ length chunk)
                    chunk' = (\Dataset.STLCExample {..} -> (exInputIds, exTargetIds)) <$> chunk
                    (inputIds, targetIds) = unzip chunk'
                    maxInputLength' = min configMaxInputLength (foldr (max . length) 0 inputIds)
                    encoderSeqDim = SNoName :&: SUncheckedSize (fromIntegral maxInputLength')
                    maxTargetLength' = min configMaxTargetLength (foldr (max . length) 0 targetIds)
                    decoderSeqDim = SNoName :&: SUncheckedSize (fromIntegral maxTargetLength')
                 in (,)
                      <$> ( (,)
                              <$> mkT5Input batchDim encoderSeqDim device inputIds
                              <*> mkT5Input batchDim decoderSeqDim device targetIds
                          )
                      <*> pure chunk
           in bufferedCollate (P.bounded configBufferSize) configMaxBatchSize collateFn

    tokenizerSem <- MSem.new (1 :: Int)
    let tokenize input = MSem.with tokenizerSem $ do
          encoding <- Tokenizers.encode tokenizer input
          Tokenizers.getIDs encoding
        detokenize ids = MSem.with tokenizerSem $ do
          Tokenizers.decode tokenizer ids

    let trainingData =
          Dataset.STLCData
            { name = "training",
              seeds =
                Set.fromList
                  . List.take configNumTrainingExamples
                  $ Seed.from <$> List.iterate (+ 1) (minBound :: Word64),
              targetNfSteps = Set.fromList [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39],
              maxInputLength = configMaxInputLength,
              maxTargetLength = configMaxTargetLength,
              tokenize,
              detokenize
            }

        evaluationData =
          Dataset.STLCData
            { name = "evaluation",
              seeds =
                Set.fromList
                  . List.take configNumEvaluationExamples
                  $ Seed.from <$> List.iterate (\s -> s - 1) (maxBound :: Word64),
              targetNfSteps = Set.fromList [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
              maxInputLength = configMaxInputLength,
              maxTargetLength = configMaxTargetLength,
              tokenize,
              detokenize
            }

        streamingState = (datasetOpts configNumWorkers) {shuffle = Shuffle (mkStdGen (fromIntegral configSeed))}

    -- create an Adam optimizer from the model
    optim <- liftIO $ mkAdam defaultAdamOptions {learningRate = configLearningRate} model

    let -- one epoch of training and evaluation
        step (streamingState', g2) epoch = do
          -- let learningRate = learningRateSchedule epoch
          -- setLearningRate optim learningRate

          -- helper function for streaming
          let go streamingState'' data' closure' = do
                (stream, shuffle) <- P.lift $ streamFromMap streamingState'' data'
                r <- P.lift $ do
                  batchedStream <- collate' stream
                  lift $ closure' batchedStream
                case r of
                  Left g -> pure (g, shuffle)
                  Right (monitor', g) -> do
                    P.yield (monitor' epoch)
                    pure (g, shuffle)

          -- train for one epoch on the training set
          (g3, shuffle) <- go streamingState' trainingData $ \batchedStream -> do
            r <- train optim trainingModelSpec (fst <$> batchedStream) g2
            pure $ (\(loss, g) -> (TrainingLossMonitor (fromTensor loss), g)) <$> r

          -- evaluate on the evaluation set
          (g4, _sample) <- go streamingState' {shuffle = Sequential} evaluationData $ \batchedStream -> do
            stateDict' <- getStateDict optim
            model' <- flip evalStateT stateDict' $ fromStateDict evaluationModelSpec mempty
            r <- eval model' (fst <$> batchedStream) g3
            stateDictToFile stateDict' configModelSavePath
            pure $ (\(loss, g) -> (EvaluationLossMonitor (fromTensor loss), g)) <$> r

          (g5, _sample) <- go streamingState' {shuffle = Sequential} evaluationData $ \batchedStream -> do
            stateDict' <- getStateDict optim
            Model.NeuralInterpreter t5 <- flip evalStateT stateDict' $ fromStateDict evaluationModelSpec mempty
            let step' ((targets, predictions), g) (encoderInput, decoderInput) = do
                  let postProcess =
                        Text.unpack
                          . Text.replace "<unk>" "\\"
                          . Text.replace "<pad>" mempty
                          . Text.pack

                  let decoderIds :: [[Int]] = fromTensor decoderInput
                  targets' <- traverse ((postProcess <$>) . Tokenizers.decode tokenizer) decoderIds

                  let input = SimplifiedEncoderDecoderTransformerInput' encoderInput
                      [Dim _ batchSize, Dim _ _] = getDims encoderInput
                      batchDim = SNoName :&: SUncheckedSize batchSize

                  (SimplifiedEncoderDecoderTransformerOutput' encoderOutput paddingMask, g') <- forward t5 input g

                  x <-
                    SimplifiedEncoderDecoderTransformerGenerationInput
                      <$> mkTransformerInput
                        t5PadTokenId
                        batchDim
                        (SNoName :&: SUncheckedSize 0)
                        device
                        []
                      <*> pure encoderOutput
                      <*> pure paddingMask

                  us <- sOnes $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device (SDataType SInt64) (SShape $ batchDim :|: SNil)

                  ((SimplifiedEncoderDecoderTransformerGenerationInput decoderInput' _ _, g''), _us) <-
                    flip runStateT us $
                      decode
                        ( \input@(SimplifiedEncoderDecoderTransformerGenerationInput decoderInput' _ _) g -> do
                            let [Dim _ _, Dim _ seqLen] = getDims decoderInput'
                            unfinishedSequences <- get
                            b <- allSequencesFinished unfinishedSequences
                            if b || seqLen >= fromIntegral configMaxTargetLength
                              then pure Nothing
                              else do
                                (output, g') <- forward t5 input g
                                input' <- (sedtOutputToInput . prepNext %~ greedyNextTokens t5PadTokenId t5EOSTokenId) output
                                pure $ Just (input', g')
                        )
                        x
                        g'

                  let decoderIds' :: [[Int]] = fromTensor decoderInput'
                  predictions' <- traverse ((postProcess <$>) . Tokenizers.decode tokenizer) decoderIds'

                  pure ((targets <> targets', predictions <> predictions'), g'')

                init'' = pure (mempty, g4)

                done' ((targets, predictions), g5) = do
                  let exactMatchAccuracy =
                        let xs = zip targets predictions
                         in (fromIntegral $ length . filter (uncurry (==)) $ xs) / (fromIntegral $ length xs)
                  pure $ Right (PredictionsMonitor targets predictions exactMatchAccuracy, g5)

            P.foldM step' init'' done' $ P.enumerate (fst <$> batchedStream)

          pure (streamingState' {shuffle}, g5)

    let init' = pure (streamingState, g1)
        done (_streamingState', _g) = pure ()

    flip runContT pure . P.runEffect $
      P.foldM step init' done (P.each [1 .. configNumEpochs]) P.>-> monitor

main :: IO ()
main = do
  conf@Config {..} <- Opts.execParser opts

  let device = SUncheckedDevice configDevice

  case configModelArchitecture of
    T5Small -> do
      let trainingModelSpec = Model.NeuralInterpreter $ t5SmallSpec SWithLMHead (SGradient SWithGradient) device SWithDropout
          evaluationModelSpec = Model.NeuralInterpreter $ t5SmallSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
      go conf device trainingModelSpec evaluationModelSpec
    T5Large -> do
      let trainingModelSpec = Model.NeuralInterpreter $ t5LargeSpec SWithLMHead (SGradient SWithGradient) device SWithDropout
          evaluationModelSpec = Model.NeuralInterpreter $ t5LargeSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
      go conf device trainingModelSpec evaluationModelSpec
    T5ThreeB -> do
      let trainingModelSpec = Model.NeuralInterpreter $ t5ThreeBSpec SWithLMHead (SGradient SWithGradient) device SWithDropout
          evaluationModelSpec = Model.NeuralInterpreter $ t5ThreeBSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
      go conf device trainingModelSpec evaluationModelSpec
    BARTBase -> do
      let trainingModelSpec = Model.NeuralInterpreter $ bartBaseSpec SWithLMHead (SGradient SWithGradient) device SWithDropout
          evaluationModelSpec = Model.NeuralInterpreter $ bartBaseSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
      go conf device trainingModelSpec evaluationModelSpec
    BARTLarge -> do
      let trainingModelSpec = Model.NeuralInterpreter $ bartLargeSpec SWithLMHead (SGradient SWithGradient) device SWithDropout
          evaluationModelSpec = Model.NeuralInterpreter $ bartLargeSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
      go conf device trainingModelSpec evaluationModelSpec
