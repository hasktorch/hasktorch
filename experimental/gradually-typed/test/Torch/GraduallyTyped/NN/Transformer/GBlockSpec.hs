{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GBlockSpec where

import Control.Monad.State (evalStateT)
import qualified Data.Map as Map
import Torch.GraduallyTyped

testEncoderBlock :: IO _
testEncoderBlock = do
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
  let spec = NamedModel "block." $ encoderBlockSpec ST5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim SWithDropout dropoutP eps
  (encoderBlock, g') <- initialize spec g
  encoderBlock' <- flip evalStateT Map.empty $ do
    toStateDict mempty encoderBlock
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  query <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
  attentionBias <- sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward encoderBlock' (query, attentionBias) g'
  pure output

testDecoderBlock :: IO _
testDecoderBlock = do
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
  let spec = NamedModel "block." $ decoderBlockSpec SByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim SWithDropout dropoutP eps
  (decoderBlock, g') <- initialize spec g
  decoderBlock' <- flip evalStateT Map.empty $ do
    toStateDict mempty decoderBlock
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      decoderSeqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  query <- sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: queryEmbedDim :|: SNil)
  key <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
  decoderAttentionBias <- sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
  crossAttentionBias <- sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoderBlock' (query, key, decoderAttentionBias, crossAttentionBias) g'
  pure output
