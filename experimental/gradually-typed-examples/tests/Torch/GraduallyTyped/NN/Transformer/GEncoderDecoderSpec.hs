{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GEncoderDecoderSpec where

import Torch.GraduallyTyped

testEncoderDecoderTransformer :: IO _
testEncoderDecoderTransformer = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      inputEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      posEncDim = SName @"*" :&: SSize @32
      vocabDim = SName @"*" :&: SSize @32128
      dropoutP = 0
      eps = 1e-6
  g <- sMkGenerator device 0
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  edtInput <- sOnes' (SDataType SInt64) (SShape $ batchDim :|: seqDim :|: SNil)
  edtAttentionMask <- sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  edtDecoderInput <- sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
  edtDecoderAttentionMask <- sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
  edtCrossAttentionMask <- sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (t5Output, g'') <- do
    let spec = NamedModel "t5." $ encoderDecoderTransformerSpec ST5 SWithLMHead (SNat @11) (SNat @7) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim SWithDropout dropoutP eps
    (t5, g') <- initialize spec g
    edtPos <- sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
    edtDecoderPos <- sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
    forward t5 EncoderDecoderTransformerInput {..} g'
  (bartOutput, g'''') <- do
    let spec = NamedModel "bart." $ encoderDecoderTransformerSpec SBART SWithLMHead (SNat @17) (SNat @13) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim SWithDropout dropoutP eps
    (bart, g''') <- initialize spec g''
    edtPos <- sOnes' (SDataType SInt64) (SShape $ seqDim :|: SNil)
    edtDecoderPos <- sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
    forward bart EncoderDecoderTransformerInput {..} g'''
  pure ((t5Output, bartOutput), g'''')
