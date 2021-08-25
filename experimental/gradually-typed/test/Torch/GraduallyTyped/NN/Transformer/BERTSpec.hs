{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.BERTSpec where

import Control.Monad.State (evalStateT)
import Data.Singletons (SingKind (fromSing))
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers
import Torch.GraduallyTyped

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/bert-base-uncased-tokenizer.json"

testForwardBERTBaseUncased :: IO ()
testForwardBERTBaseUncased =
  do
    stateDict <- stateDictFromFile "/tmp/bert-base-uncased-state-dict.pt"

    let device = SDevice SCPU

    let spec = bertBaseUnchasedSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
    model <- flip evalStateT stateDict $ fromStateDict spec mempty

    g <- sMkGenerator device 0

    ids <- withTokenizer $ \tokenizer -> do
      encoding <- Tokenizers.encode tokenizer "[CLS] the capital of france is [MASK]. [SEP]"
      Tokenizers.getIDs encoding
    let batchDim = SName @"*" :&: SSize @1
        seqSize = SUncheckedSize . fromIntegral $ length ids
        seqDim = SName @"*" :&: seqSize

    input <- do
      inputType <-
        sZeros $
          TensorSpec
            (SGradient SWithoutGradient)
            (SLayout SDense)
            device
            (SDataType SInt64)
            (SShape $ batchDim :|: seqDim :|: SNil)
      SimplifiedEncoderOnlyTransformerInput
        <$> mkBERTInput batchDim seqDim device [ids]
        <*> pure inputType

    (SimplifiedEncoderOnlyTransformerOutput {..}, _) <- forward model input g

    encoderOutput :: [[[Float]]] <-
      fromTensor
        <$> sCheckedShape
          ( SShape $
              SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ batchDim)
                :|: SName @"*" :&: seqSize
                :|: SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ bertBaseUncasedVocabDim)
                :|: SNil
          )
          seotOutput
    let firstLMHeadLogits = do
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLMHeadLogits' = [-6.4346, -6.4063, -6.4097, -14.0119, -14.7240, -14.2120, -9.6561, -10.3125, -9.7459]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits' firstLMHeadLogits
