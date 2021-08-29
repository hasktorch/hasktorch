{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GLMHeadSpec where

import Control.Monad.State (evalStateT)
import qualified Data.Map as Map
import Torch.GraduallyTyped

testLMHead :: IO _
testLMHead = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      inputEmbedDim = SName @"*" :&: SSize @512
      vocabDim = SName @"*" :&: SSize @30522
      eps = 1e-6
  g <- sMkGenerator device 0
  let spec = NamedModel "lmHead." $ lmHeadSpec SBART gradient device dataType inputEmbedDim vocabDim eps
  (lmHead, g') <- initialize spec g
  lmHead' <- flip evalStateT Map.empty $ do
    toStateDict mempty lmHead
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  input <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: inputEmbedDim :|: SNil)
  (output, _) <- forward lmHead' input g'
  pure output
