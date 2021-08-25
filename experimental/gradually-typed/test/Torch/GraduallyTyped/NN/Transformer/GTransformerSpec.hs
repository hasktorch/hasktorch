{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GTransformerSpec where

import Control.Monad.State (evalStateT)
import qualified Data.Map as Map
import Torch.GraduallyTyped

testEncoder :: IO _
testEncoder = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      inputEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      posEncDim = SName @"*" :&: SSize @32
      dropoutP = 0
      eps = 1e-6
  g <- sMkGenerator device 0
  let spec = NamedModel "encoder." $ transformerEncoderSpec ST5 (SNat @10) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim SWithDropout dropoutP eps
  (encoder, g') <- initialize spec g
  encoder' <- flip evalStateT Map.empty $ do
    toStateDict mempty encoder
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  input <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: inputEmbedDim :|: SNil)
  relPos <- sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  attentionMask <- sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward encoder' (input, relPos, attentionMask) g'
  pure output

testDecoder :: IO _
testDecoder = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      decoderInputEmbedDim = SName @"*" :&: SSize @512
      encoderOutputEmbedDim = decoderInputEmbedDim
      ffnDim = SName @"*" :&: SSize @2048
      posEncDim = SName @"*" :&: SSize @32
      dropoutP = 0
      eps = 1e-6
  g <- sMkGenerator device 0
  let spec = NamedModel "decoder." $ transformerDecoderSpec SBART (SNat @10) gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim SWithDropout dropoutP eps
  (decoder, g') <- initialize spec g
  decoder' <- flip evalStateT Map.empty $ do
    toStateDict mempty decoder
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  decoderInput <- sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: decoderInputEmbedDim :|: SNil)
  encoderOutput <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: encoderOutputEmbedDim :|: SNil)
  decoderPos <- sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
  decoderAttentionMask <- sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
  crossAttentionMask <- sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoder' (decoderInput, encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask) g'
  pure output
