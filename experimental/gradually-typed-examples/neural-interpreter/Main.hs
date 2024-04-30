{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

module Main where

import qualified Control.Concurrent.MSem as MSem
import Control.Lens ((%~))
import Control.Monad.Cont (runContT)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.State (evalStateT, get, runStateT)
import Control.Monad.Trans (MonadTrans (..))
import qualified Data.List as List
import Data.Monoid (Sum (Sum))
import qualified Data.Set as Set
import qualified Data.Text as Text
import Data.Word (Word64)
import qualified Dataset (STLCData (..), STLCExample (..))
import qualified Hedgehog.Internal.Seed as Seed
import qualified Model (NeuralInterpreter (..))
import qualified Monitor
import qualified Options.Applicative as Opts
import qualified Opts
import Pipes ((>->))
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Prelude as P hiding (toHandle)
import qualified Pipes.Safe as P
import qualified Pipes.Safe.Prelude as P
import qualified System.IO as IO
import System.Random (mkStdGen)
import qualified Tokenizers
import Torch.GraduallyTyped

go :: _ => _
go config@Opts.Config {..} device trainingModelSpec evaluationModelSpec =
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
              targetNfSteps = configTrainingNFReductionSteps,
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
              targetNfSteps = configEvaluationNFReductionSteps,
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
          -- helper function for streaming
          let go' streamingState'' data' closure' = do
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
          (g3, shuffle) <- go' streamingState' trainingData $ \batchedStream -> do
            r <- train optim trainingModelSpec (fst <$> batchedStream) g2
            pure $ (\(loss, g) -> (Monitor.TrainingMonitor (fromTensor loss), g)) <$> r

          -- evaluate on the evaluation set
          (g4, _sample) <- go' streamingState' {shuffle = Sequential} evaluationData $ \batchedStream -> do
            stateDict' <- liftIO $ getStateDict optim
            model'@(Model.NeuralInterpreter transformer) <- flip evalStateT stateDict' $ fromStateDict evaluationModelSpec mempty

            let step' (((loss, targets, predictions, examples), _), g) (((encoderInput, decoderInput), examples'), iter) = do
                  (loss', g') <- forward model' (encoderInput, decoderInput) g
                  loss'' <- sCheckedShape (SShape SNil) =<< sCheckedGradient (SGradient SWithoutGradient) loss'
                  let loss''' :: Float = fromTensor loss''

                  let postProcess =
                        Text.unpack
                          . ( case configModelArchitecture of
                                Opts.T5Small -> Text.replace "<unk>" "\\"
                                Opts.T5Base -> Text.replace "<unk>" "\\"
                                Opts.T5Large -> Text.replace "<unk>" "\\"
                                Opts.T5ThreeB -> Text.replace "<unk>" "\\"
                                Opts.BARTBase -> id
                                Opts.BARTLarge -> id
                            )
                          . Text.replace "<pad>" mempty
                          . Text.pack

                  let decoderIds :: [[Int]] = fromTensor decoderInput
                  targets' <- liftIO $ traverse ((postProcess <$>) . Tokenizers.decode tokenizer) decoderIds

                  let input = SimplifiedEncoderDecoderTransformerInput' encoderInput
                      [Dim _ batchSize, Dim _ _] = getDims encoderInput
                      batchDim = SNoName :&: SUncheckedSize batchSize

                  (SimplifiedEncoderDecoderTransformerOutput' encoderOutput paddingMask, g'') <- forward transformer input g'

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

                  ((SimplifiedEncoderDecoderTransformerGenerationInput decoderInput' _ _, g'''), _us) <-
                    flip runStateT us $
                      decode
                        ( \input'@(SimplifiedEncoderDecoderTransformerGenerationInput decoderInput' _ _) g -> do
                            let [Dim _ _, Dim _ seqLen] = getDims decoderInput'
                            unfinishedSequences <- get
                            b <- allSequencesFinished unfinishedSequences
                            if b || seqLen >= fromIntegral configMaxTargetLength
                              then pure Nothing
                              else do
                                (output, g') <- forward transformer input' g
                                input'' <- (sedtOutputToInput . prepNext %~ greedyNextTokens t5PadTokenId t5EOSTokenId) output
                                pure $ Just (input'', g')
                        )
                        x
                        g''

                  let decoderIds' :: [[Int]] = fromTensor decoderInput'
                  predictions' <- liftIO $ traverse ((postProcess <$>) . Tokenizers.decode tokenizer) decoderIds'

                  pure (((loss <> Sum loss''', targets <> targets', predictions <> predictions', examples <> examples'), iter), g''')

                init'' = pure ((mempty, 0), g3)

                done' (((Sum loss, targets, predictions, examples), iter), g4) = do
                  let avgLoss = loss / fromIntegral iter
                      exactMatchAccuracy =
                        let xs = zip targets predictions
                         in fromIntegral (length . filter (uncurry (==)) $ xs) / fromIntegral (length xs)
                  pure $ Right (Monitor.EvaluationMonitor avgLoss targets predictions exactMatchAccuracy examples, g4)

            P.foldM step' init'' done' $ P.zip (P.enumerate batchedStream) (P.each [0 :: Int ..])

          pure (streamingState' {shuffle}, g4)

    let init' = pure (streamingState, g1)
        done (_streamingState', _g) = pure ()

    let outputStream = do
          P.yield (Monitor.ConfigMonitor config)
          P.foldM step init' done (P.each [1 .. configNumEpochs])

    P.runSafeT $
      P.withFile configOutputPath IO.WriteMode $ \h ->
        flip runContT pure . P.runEffect $
          outputStream >-> Monitor.monitor configNumEpochs h

main :: IO ()
main = do
  conf@Opts.Config {..} <- Opts.execParser Opts.opts

  let device = SUncheckedDevice configDevice

  let trainingModelSpec :: _
      trainingModelSpec transformerSpec = Model.NeuralInterpreter $ transformerSpec SWithLMHead (SGradient SWithGradient) device SWithDropout
      evaluationModelSpec :: _
      evaluationModelSpec transformerSpec = Model.NeuralInterpreter $ transformerSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout

  case configModelArchitecture of
    Opts.T5Small -> go conf device (trainingModelSpec t5SmallSpec) (evaluationModelSpec t5SmallSpec)
    Opts.T5Base -> go conf device (trainingModelSpec t5BaseSpec) (evaluationModelSpec t5BaseSpec)
    Opts.T5Large -> go conf device (trainingModelSpec t5LargeSpec) (evaluationModelSpec t5LargeSpec)
    Opts.T5ThreeB -> go conf device (trainingModelSpec t5ThreeBSpec) (evaluationModelSpec t5ThreeBSpec)
    Opts.BARTBase -> go conf device (trainingModelSpec bartBaseSpec) (evaluationModelSpec bartBaseSpec)
    Opts.BARTLarge -> go conf device (trainingModelSpec bartLargeSpec) (evaluationModelSpec bartLargeSpec)
