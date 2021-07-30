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

    let device = SDevice (SCUDA @0)

    let spec = bertBaseUnchasedSpec SWithLMHead (SGradient SWithGradient) device
    GBERTModel {..} <- flip evalStateT stateDict $ fromStateDict spec mempty

    let g = sMkGenerator device 0

    ids <- withTokenizer $ \tokenizer -> do
      encoding <- Tokenizers.encode tokenizer "[CLS] the capital of france is [MASK]. [SEP]"
      Tokenizers.getIDs encoding
    let seqSize = SUncheckedSize . fromIntegral $ length ids

    input <-
      mkBERTInput
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
          sArangeNaturals
            (SGradient SWithoutGradient)
            (SLayout SDense)
            device
            (SDataType SInt64)
            seqSize
        paddingMask = mkBERTPaddingMask input
    attentionMask <- mkTransformerAttentionMask bertDataType bertAttentionMaskBias paddingMask

    let eotInput = EncoderOnlyTransformerInput input inputType pos attentionMask
    (EncoderOnlyTransformerOutput {..}, _) <- forward bertModel eotInput g

    encoderOutput :: [[[Float]]] <-
      fromTensor
        <$> sCheckedShape
          ( SShape $
              SName @"*" :&: SUncheckedSize 1
                :|: SName @"*" :&: seqSize
                :|: SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ bertBaseUncasedVocabDim)
                :|: SNil
          )
          eoEncoderOutput
    let firstLMHeadLogits = do
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLMHeadLogits' = [-6.4346, -6.4063, -6.4097, -14.0119, -14.7240, -14.2120, -9.6561, -10.3125, -9.7459]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits' firstLMHeadLogits
