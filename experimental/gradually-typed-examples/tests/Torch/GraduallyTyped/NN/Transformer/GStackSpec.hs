{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GStackSpec where

import Control.Monad.State (evalStateT)
import qualified Data.Map as Map
import Torch.GraduallyTyped

testEncoderStack :: IO _
testEncoderStack = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      dropoutP = 0
      eps = 1e-6
  g <- sMkGenerator device 0
  let spec = NamedModel "stack." $ encoderStackSpec ST5 (SNat @2) gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim SWithDropout dropoutP eps
  (encoderStack, g') <- initialize spec g
  encoderStack' <- flip evalStateT Map.empty $ do
    toStateDict mempty encoderStack
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  query <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
  attentionBias <- sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward encoderStack' (query, attentionBias) g'
  pure output

testDecoderStack :: IO _
testDecoderStack = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      keyEmbedDim = queryEmbedDim
      ffnDim = SName @"*" :&: SSize @2048
      dropoutP = 0
      eps = 1e-6
  g <- sMkGenerator device 0
  let spec = NamedModel "stack." $ decoderStackSpec ST5 (SNat @2) gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim SWithDropout dropoutP eps
  (decoderStack, g') <- initialize spec g
  decoderStack' <- flip evalStateT Map.empty $ do
    toStateDict mempty decoderStack
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      decoderSeqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  query <- sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: queryEmbedDim :|: SNil)
  key <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
  attentionBias <- sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
  crossAttentionBias <- sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoderStack' (query, key, attentionBias, crossAttentionBias) g'
  pure output
