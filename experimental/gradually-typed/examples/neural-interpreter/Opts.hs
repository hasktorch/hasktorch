{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Opts where

import Control.Applicative ((<**>))
import Data.Aeson.TH (defaultOptions, deriveJSON)
import Data.Functor ((<&>))
import Data.Int (Int16)
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Word (Word64)
import GHC.Generics (Generic)
import qualified Options.Applicative as Opts
import Torch.GraduallyTyped

data ModelArchitecture = T5Small | T5Base | T5Large | T5ThreeB | BARTBase | BARTLarge
  deriving stock (Eq, Ord, Show, Generic)

data Config = Config
  { configNumTrainingExamples :: Int,
    configTrainingNFReductionSteps :: Maybe (Set Int),
    configNumEvaluationExamples :: Int,
    configEvaluationNFReductionSteps :: Maybe (Set Int),
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
    configLearningRate :: Double,
    configOutputPath :: FilePath
  }
  deriving stock (Eq, Ord, Show, Generic)

$(deriveJSON defaultOptions ''ModelArchitecture)
$(deriveJSON defaultOptions ''DeviceType)
$(deriveJSON defaultOptions ''Config)

config :: Opts.Parser Config
config =
  Config
    <$> Opts.option
      Opts.auto
      (Opts.long "num-training-examples" <> Opts.short 'n' <> Opts.value 65536 <> Opts.showDefault <> Opts.help "Number of training examples")
    <*> Opts.optional
      ( Opts.option
          (Opts.auto <&> Set.fromList)
          ( Opts.long "training-nf-reduction-steps" <> Opts.help "Number of normal-form reduction steps allowed during training. If not specified, all steps are allowed. Say \"[1,2,3]\" to allow only 1, 2, and 3 steps."
          )
      )
    <*> Opts.option
      Opts.auto
      (Opts.long "num-evaluation-examples" <> Opts.value 4096 <> Opts.showDefault <> Opts.help "Number of evaluation examples")
    <*> Opts.optional
      ( Opts.option
          (Opts.auto <&> Set.fromList)
          ( Opts.long "evaluation-nf-reduction-steps" <> Opts.help "Number of normal-form reduction steps allowed during evaluation. If not specified, all steps are allowed. Say \"[4,5]\" to allow only 4 and 5 steps."
          )
      )
    <*> Opts.option
      Opts.auto
      (Opts.long "num-epochs" <> Opts.short 'e' <> Opts.value 64 <> Opts.showDefault <> Opts.help "Number of epochs")
    <*> Opts.option
      Opts.auto
      (Opts.long "num-workers" <> Opts.short 'w' <> Opts.value 4 <> Opts.showDefault <> Opts.help "Number of workers for data loading")
    <*> Opts.option
      Opts.auto
      (Opts.long "buffer-size" <> Opts.value 32 <> Opts.showDefault <> Opts.help "Buffer size for streaming")
    <*> Opts.option
      ( Opts.auto >>= \case
          -1 -> pure CPU
          i -> pure (CUDA i)
      )
      (Opts.long "device" <> Opts.short 'd' <> Opts.value CPU <> Opts.showDefault <> Opts.help "Device. Say -1 for CPU, 0 for CUDA 0, 1 for CUDA 1, etc.")
    <*> Opts.option
      ( Opts.auto >>= \case
          (0 :: Int) -> pure T5Small
          1 -> pure T5Base
          2 -> pure T5Large
          3 -> pure T5ThreeB
          4 -> pure BARTBase
          5 -> pure BARTLarge
          _ -> Opts.readerError "Invalid model architecture"
      )
      (Opts.long "model-architecture" <> Opts.short 'a' <> Opts.value T5Small <> Opts.showDefault <> Opts.help "Model architecture. Say 0 for T5-Small, 1 for T5-Base, 2 for T5-Large, 3 for T5-ThreeB, 4 for BART-Base, 5 for BART-Large")
    <*> Opts.optional
      (Opts.strOption (Opts.long "model-pretrained-path" <> Opts.short 'p' <> Opts.metavar "PFILE" <> Opts.help "Load pretrained model from PFILE. If not specified, a new model will be trained"))
    <*> Opts.strOption
      (Opts.long "model-save-path" <> Opts.short 'm' <> Opts.metavar "MFILE" <> Opts.help "Write model to MFILE")
    <*> Opts.strOption
      (Opts.long "tokenizer-path" <> Opts.short 't' <> Opts.metavar "TFILE" <> Opts.help "Load tokenizer from TFILE")
    <*> Opts.option
      Opts.auto
      (Opts.long "seed" <> Opts.short 's' <> Opts.value 31415 <> Opts.showDefault <> Opts.help "Seed")
    <*> Opts.option
      Opts.auto
      (Opts.long "max-input-length" <> Opts.value 256 <> Opts.showDefault <> Opts.help "Maximum input length in tokens")
    <*> Opts.option
      Opts.auto
      (Opts.long "max-target-length" <> Opts.value 128 <> Opts.showDefault <> Opts.help "Maximum target length in tokens")
    <*> Opts.option
      Opts.auto
      (Opts.long "max-batch-size" <> Opts.short 'b' <> Opts.value 24 <> Opts.showDefault <> Opts.help "Maximum batch size")
    <*> Opts.option
      Opts.auto
      (Opts.long "learning-rate" <> Opts.short 'l' <> Opts.value 0.0001 <> Opts.showDefault <> Opts.help "Learning rate")
    <*> Opts.strOption
      (Opts.long "output-path" <> Opts.short 'o' <> Opts.metavar "OFILE" <> Opts.help "Write output to OFILE")

opts :: Opts.ParserInfo Config
opts = Opts.info (config <**> Opts.helper) (Opts.fullDesc <> Opts.progDesc "Train a neural interpreter model on a synthetic dataset")
