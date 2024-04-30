{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GEncoderOnlySpec where

import Torch.GraduallyTyped

testEncoderOnlyTransformer :: IO _
testEncoderOnlyTransformer = do
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
      typeVocabDim = SName @"*" :&: SSize @2
      dropoutP = 0
      eps = 1e-6
  g <- sMkGenerator device 0
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  eotInput <- sOnes' (SDataType SInt64) (SShape $ batchDim :|: seqDim :|: SNil)
  let eotInputType = eotInput
  eotAttentionMask <- sOnes' dataType (SShape $ batchDim :|: seqDim :|: seqDim :|: SNil)
  (bertOutput, g'') <- do
    let spec = NamedModel "bert." $ encoderOnlyTransformerSpec SBERT SWithLMHead (SNat @17) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim SWithDropout dropoutP eps
    (bert, g') <- initialize spec g
    eotPos <- sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: SNil)
    forward bert EncoderOnlyTransformerInput {..} g'
  (robertaOutput, g'''') <- do
    let spec = NamedModel "roberta." $ encoderOnlyTransformerSpec SRoBERTa SWithLMHead (SNat @19) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim SWithDropout dropoutP eps
    (roberta, g''') <- initialize spec g''
    eotPos <- sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: SNil)
    forward roberta EncoderOnlyTransformerInput {..} g'''
  pure ((bertOutput, robertaOutput), g'''')
