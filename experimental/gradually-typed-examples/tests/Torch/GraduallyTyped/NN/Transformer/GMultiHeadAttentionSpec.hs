{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GMultiHeadAttentionSpec where

import Control.Monad.State (evalStateT)
import qualified Data.Map as Map
import Torch.GraduallyTyped

testMHA :: IO _
testMHA = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @2
      headEmbedDim = SName @"*" :&: SSize @2
      embedDim = SName @"*" :&: SSize @4
      queryEmbedDim = SName @"*" :&: SSize @3
      keyEmbedDim = SName @"*" :&: SSize @5
      valueEmbedDim = SName @"*" :&: SSize @7
      dropoutP = 0
  g <- sMkGenerator device 0
  let spec = NamedModel "mha." $ multiHeadAttentionSpec SByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim SWithDropout dropoutP
  (mha, g') <- initialize spec g
  mha' <- flip evalStateT Map.empty $ do
    toStateDict mempty mha
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @2
      seqDim = SName @"*" :&: SSize @1
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  query <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
  key <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
  value <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: valueEmbedDim :|: SNil)
  attentionBias <- sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward mha' (query, key, value, attentionBias) g'
  pure output
