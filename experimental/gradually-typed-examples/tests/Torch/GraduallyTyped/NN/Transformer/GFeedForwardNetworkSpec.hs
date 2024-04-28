{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GFeedForwardNetworkSpec where

import Control.Monad.State (evalStateT)
import qualified Data.Map as Map
import Torch.GraduallyTyped

testFFN :: IO _
testFFN = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      ffnDim = SName @"*" :&: SSize @2
      queryEmbedDim = SName @"*" :&: SSize @3
      dropoutP = 0
      eps = 1e-6
  g <- sMkGenerator device 0
  let spec = NamedModel "ffn." $ transformerFeedForwardNetworkSpec SByT5 gradient device dataType queryEmbedDim ffnDim SWithDropout dropoutP eps
  (ffn, g') <- initialize spec g
  ffn' <- flip evalStateT Map.empty $ do
    toStateDict mempty ffn
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @2
      seqDim = SName @"*" :&: SSize @1
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  query <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
  (output, _) <- forward ffn' query g'
  pure output
