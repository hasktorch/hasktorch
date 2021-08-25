{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTaSpec where

import Control.Monad.State (evalStateT)
import Data.Singletons (SingKind (fromSing))
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers
import Torch.GraduallyTyped

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/roberta-base-tokenizer.json"

testForwardRoBERTaBase :: IO ()
testForwardRoBERTaBase =
  do
    stateDict <- stateDictFromFile "/tmp/roberta-base-state-dict.pt"

    let device = SDevice SCPU

    let spec = robertaBaseSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
    model <- flip evalStateT stateDict $ fromStateDict spec mempty

    g <- sMkGenerator device 0

    ids <- withTokenizer $ \tokenizer -> do
      encoding <- Tokenizers.encode tokenizer "<s>The capital of France is [MASK].</s>"
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
        <$> mkRoBERTaInput batchDim seqDim device [ids]
        <*> pure inputType

    (SimplifiedEncoderOnlyTransformerOutput {..}, _) <- forward model input g

    encoderOutput :: [[[Float]]] <-
      fromTensor
        <$> sCheckedShape
          ( SShape $
              SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ batchDim)
                :|: seqDim
                :|: SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ robertaBaseVocabDim)
                :|: SNil
          )
          seotOutput
    let firstLMHeadLogits = do
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLMHeadLogits' = [32.5267, -4.5318, 21.4297, 7.9570, -2.7508, 21.1128, -2.8331, -4.1595, 10.6294]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits' firstLMHeadLogits
