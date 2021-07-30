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

    let spec = robertaBaseSpec SWithLMHead (SGradient SWithoutGradient) device
    GRoBERTaModel {..} <- flip evalStateT stateDict $ fromStateDict spec mempty

    let g = sMkGenerator device 0

    ids <- withTokenizer $ \tokenizer -> do
      encoding <- Tokenizers.encode tokenizer "<s>The capital of France is [MASK].</s>"
      Tokenizers.getIDs encoding
    let seqSize = SUncheckedSize . fromIntegral $ length ids

    input <-
      mkRoBERTaInput
        (SName @"*" :&: SSize @1)
        (SName @"*" :&: seqSize)
        device
        [ids]
    let inputType =
          sZeros $
            TensorSpec
              (SGradient SWithoutGradient)
              (SLayout SDense)
              device
              (SDataType SInt64)
              (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: seqSize :|: SNil)
        pos =
          flip addScalar (2 :: Int) $
            sArangeNaturals
              (SGradient SWithoutGradient)
              (SLayout SDense)
              device
              (SDataType SInt64)
              seqSize
        paddingMask = mkRoBERTaPaddingMask input
    attentionMask <- mkTransformerAttentionMask robertaDataType robertaAttentionMaskBias paddingMask

    let eotInput = EncoderOnlyTransformerInput input inputType pos attentionMask
    (EncoderOnlyTransformerOutput {..}, _) <- forward robertaModel eotInput g

    encoderOutput :: [[[Float]]] <-
      fromTensor
        <$> sCheckedShape
          ( SShape $
              SName @"*" :&: SUncheckedSize 1
                :|: SName @"*" :&: seqSize
                :|: SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ robertaBaseVocabDim)
                :|: SNil
          )
          eoEncoderOutput
    let firstLMHeadLogits = do
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLMHeadLogits' = [32.5267, -4.5318, 21.4297, 7.9570, -2.7508, 21.1128, -2.8331, -4.1595, 10.6294]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits' firstLMHeadLogits
