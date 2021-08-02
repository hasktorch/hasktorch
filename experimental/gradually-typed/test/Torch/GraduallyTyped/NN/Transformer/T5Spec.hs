{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.T5Spec where

import Control.Monad.State (evalStateT)
import Data.Singletons (SingKind (fromSing))
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers
import Torch.GraduallyTyped

testT5 :: IO _
testT5 = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      inputEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      vocabDim = SName @"*" :&: SSize @32128
  g <- sMkGenerator device 0
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
  edtInput <- sOnes' (SDataType SInt64) (SShape $ batchDim :|: seqDim :|: SNil)
  edtAttentionMask <- sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  edtDecoderInput <- sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
  edtDecoderAttentionMask <- sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
  edtCrossAttentionMask <- sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  let spec = encoderDecoderTransformerSpec ST5 SWithLMHead (SNat @4) (SNat @4) gradient device t5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim t5RelPosEncBucketDim vocabDim t5DropoutP t5Eps
  (sedtModel, g') <- initialize spec g
  (t5Output, g'') <- do
    edtPos <- sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
    edtDecoderPos <- sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
    forward sedtModel EncoderDecoderTransformerInput {..} g'
  (t5Output', g''') <-
    let sedtDecoderInputShift = ShiftRight t5BOSTokenId
        sedtPaddingMaskShift = ShiftRight 0
        sedtMkPos = MkRelPos t5RelPosEncBucketDim t5MaxDistance
        sedtMkDecoderPos = MkDecoderRelPos t5RelPosEncBucketDim t5MaxDistance
        sedtMkPaddingMask = MkTransformerPaddingMask t5PadTokenId
        sedtMkAttentionMask = MkTransformerAttentionMask t5DataType t5AttentionMaskBias
        sedtMkCrossAttentionMask = MkTransformerCrossAttentionMask t5DataType t5AttentionMaskBias
        sedtMkDecoderAttentionMask = MkTransformerDecoderAttentionMask t5DataType t5AttentionMaskBias
        model = GSimplifiedEncoderDecoderTransformer {..}
        inputs = SimplifiedEncoderDecoderTransformerInput edtInput edtDecoderInput
     in forward model inputs g''
  pure ((t5Output, t5Output'), g''')

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/t5-small-tokenizer.json"

testForwardT5Small :: IO ()
testForwardT5Small =
  do
    stateDict <- stateDictFromFile "/tmp/t5-small-state-dict.pt"

    let device = SDevice SCPU

    let spec = t5SmallSpec SWithLMHead (SGradient SWithoutGradient) device
    model <- flip evalStateT stateDict $ fromStateDict spec mempty

    g <- sMkGenerator device 0

    (encoderIds, decoderIds) <- withTokenizer $ \tokenizer -> do
      encoderEncoding <- Tokenizers.encode tokenizer "translate English to German: Studies have shown that owning a dog is good for you and your dog.</s>"
      decoderEncoding <- Tokenizers.encode tokenizer "Studien haben gezeigt, dass das Besitzen eines Hundes gut für Sie und Ihren Hund ist.</s>"
      (,) <$> Tokenizers.getIDs encoderEncoding <*> Tokenizers.getIDs decoderEncoding
    let batchDim = SName @"*" :&: SSize @1
        encoderSeqSize = SUncheckedSize . fromIntegral $ length encoderIds
        encoderSeqDim = SName @"*" :&: encoderSeqSize
        decoderSeqSize = SUncheckedSize . fromIntegral $ length decoderIds
        decoderSeqDim = SName @"*" :&: decoderSeqSize

    input <-
      SimplifiedEncoderDecoderTransformerInput
        <$> mkT5Input batchDim encoderSeqDim device [encoderIds]
        <*> mkT5Input batchDim decoderSeqDim device [decoderIds]

    (SimplifiedEncoderDecoderTransformerOutput {..}, _) <- forward model input g

    decoderOutput :: [[[Float]]] <-
      fromTensor
        <$> sCheckedShape
          ( SShape $
              SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ batchDim)
                :|: SName @"*" :&: (\case SUncheckedSize size -> SUncheckedSize $ size + 1) decoderSeqSize
                :|: SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ t5SmallVocabDim)
                :|: SNil
          )
          sedtDecoderOutput
    let firstLogits = do
          firstBatch <- take 1 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLogits' =
          [-19.3826, -10.5635, -11.4550, -26.4326, -15.4450, -14.5276, -28.7175, -14.7651, -18.2521, -14.1124, -7.4893, -12.4156, -27.9005, -11.5861, -15.9638, -24.8472, -9.6344, -12.3494]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits' firstLogits

-- testForwardByT5Small :: IO ()
-- testForwardByT5Small =
--   do
--     stateDict <- stateDictFromPretrained "/tmp/byt5-small-state-dict.pt"

--     let spec = byT5SmallSpec SWithLMHead (SGradient SWithoutGradient) (SDevice SCPU)
--     model <- flip evalStateT stateDict $ fromStateDict spec mempty

--     let g <- sMkGenerator (SDevice SCPU) 0

--     input <-
--       SimplifiedEncoderDecoderTransformerInput
--         <$> undefined
--         <*> undefined

--     (SimplifiedEncoderDecoderTransformerOutput {..}, _) <- forward model input g

--     let decoderOutput = case sedtDecoderOutput of
--           UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
--     let firstLogits = do
--           firstBatch <- take 1 decoderOutput
--           firstPositions <- take 3 firstBatch
--           take 3 firstPositions
--     let firstLogits' = undefined
--     mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits' firstLogits
