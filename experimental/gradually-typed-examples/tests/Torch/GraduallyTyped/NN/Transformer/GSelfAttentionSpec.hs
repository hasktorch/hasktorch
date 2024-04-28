{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GSelfAttentionSpec where

import Control.Monad.State (evalStateT)
import qualified Data.Map as Map
import Torch.GraduallyTyped

testSA :: IO _
testSA = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      dropoutP = 0
      eps = 1e-6
  g <- sMkGenerator device 0
  let spec = NamedModel "sa." $ selfAttentionSpec SByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim SWithDropout dropoutP eps
  (sa, g') <- initialize spec g
  sa' <- flip evalStateT Map.empty $ do
    toStateDict mempty sa
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @1
      seqDim = SName @"*" :&: SSize @4
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  query <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
  attentionBias <- sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward sa' (query, attentionBias) g'
  pure output
