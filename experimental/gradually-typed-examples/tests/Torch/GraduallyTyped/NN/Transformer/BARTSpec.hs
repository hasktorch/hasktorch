{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.BARTSpec where

import Control.Monad.State (StateT (..), get, evalStateT)
import Control.Lens ((%~))
import Data.Function (fix)
import Data.List (sortBy)
import Data.Ord (Down (..), comparing)
import Data.Singletons (SingKind (fromSing))
import qualified Tokenizers (Tokenizer, decode, encode, getIDs, withTokenizerFromConfigFile)
import Torch.GraduallyTyped

testBart :: IO _
testBart = do
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
  edtAttentionMask <- sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  edtDecoderInput <- sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
  edtDecoderAttentionMask <- sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
  edtCrossAttentionMask <- sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  let spec = encoderDecoderTransformerSpec SBART SWithLMHead (SNat @4) (SNat @4) gradient device bartDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim bartPosEncDim vocabDim SWithDropout bartDropoutP bartEps
  (sedtModel, g') <- initialize spec g
  (bartOutput, g'') <- do
    edtPos <- sOnes' (SDataType SInt64) (SShape $ seqDim :|: SNil)
    edtDecoderPos <- sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
    forward sedtModel EncoderDecoderTransformerInput {..} g'
  (bartOutput', g''') <-
    let sedtDecoderInputShift = ShiftRight bartEOSTokenId
        sedtPaddingMaskShift = ShiftRight 0
        sedtMkPos = MkAbsPosWithOffset 2
        sedtMkDecoderPos = MkAbsPosWithOffset 2
        sedtMkPaddingMask = MkTransformerPaddingMask bartPadTokenId
        sedtMkAttentionMask = MkTransformerAttentionMask bartDataType bartAttentionMaskBias
        sedtMkCrossAttentionMask = MkTransformerCrossAttentionMask bartDataType bartAttentionMaskBias
        sedtMkDecoderAttentionMask = MkTransformerDecoderAttentionMask bartDataType bartAttentionMaskBias
        model = GSimplifiedEncoderDecoderTransformer {..}
        inputs = SimplifiedEncoderDecoderTransformerInput edtInput edtDecoderInput
     in forward model inputs g''
  pure ((bartOutput, bartOutput'), g''')

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/bart-base-tokenizer.json"

testBARTAutoencoder :: String -> IO String
testBARTAutoencoder prompt = do
  stateDict <- stateDictFromFile "/tmp/bart-base-state-dict.pt"

  let device = SDevice SCPU

  let spec = bartBaseSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
  model <- flip evalStateT stateDict $ fromStateDict spec mempty

  g <- sMkGenerator device 0

  withTokenizer $ \tokenizer -> do
    specialTokens <- Tokenizers.encode tokenizer "<mask></s>"
    [maskId, eosId] <- Tokenizers.getIDs specialTokens

    promptEncoding <- Tokenizers.encode tokenizer ("<s>" ++ prompt)
    promptIds <- Tokenizers.getIDs promptEncoding
    let encoderIds = promptIds ++ [maskId, eosId]
    encoderTensor <-
      mkBARTInput
        (SName @"*" :&: SSize @1)
        (SName @"*" :&: SUncheckedSize (fromIntegral $ length encoderIds))
        device
        [encoderIds]

    let maxInputSize = 512
    outputIds <- flip fix ([], g) $ \loop (completionIds, g') -> do
      let decoderIds = promptIds ++ completionIds
          decoderSeqSize = SUncheckedSize . fromIntegral $ length decoderIds
      decoderTensor <-
        mkBARTInput
          (SName @"*" :&: SSize @1)
          (SName @"*" :&: decoderSeqSize)
          device
          [decoderIds]
      let input = SimplifiedEncoderDecoderTransformerInput encoderTensor decoderTensor

      (SimplifiedEncoderDecoderTransformerOutput {..}, g'') <- forward model input g'

      decoderOutput :: [[[Float]]] <-
        fromTensor
          <$> sCheckedShape
            ( SShape $
                SName @"*" :&: SUncheckedSize 1
                  :|: SName @"*" :&: (\case SUncheckedSize size -> SUncheckedSize $ size + 1) decoderSeqSize
                  :|: SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ bartBaseVocabDim)
                  :|: SNil
            )
            sedtDecoderOutput
      let (lastId, _lastLogit) = head $ do
            firstBatch <- take 1 decoderOutput
            let lastPosition = last firstBatch
            sortBy (comparing $ Down . snd) $ zip [0 :: Int ..] $ filter (/= 50108) lastPosition
      if lastId == eosId || length decoderIds + 1 >= maxInputSize
        then pure (promptIds ++ completionIds ++ [lastId])
        else loop (completionIds ++ [lastId], g'')

    Tokenizers.decode tokenizer outputIds


testGreedySearch :: [String] -> IO [String]
testGreedySearch xs = withTokenizer $ \tokenizer -> do

      stateDict <- stateDictFromFile "/tmp/bart-base-state-dict.pt"

      encoderIds <- traverse (\s -> Tokenizers.encode tokenizer s >>= Tokenizers.getIDs) xs

      let device = SDevice SCPU
          padTokenId = bartPadTokenId
          eosTokenId = bartEOSTokenId
          batchDim = SNoName :&: SUncheckedSize (fromIntegral $ length encoderIds)
          seqDim = SNoName :&: SUncheckedSize (fromIntegral $ min 512 (foldr (max . length) 0 encoderIds))
      
      let spec = bartBaseSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
      model <- flip evalStateT stateDict $ fromStateDict spec mempty

      g <- sMkGenerator device 0

      input <- SimplifiedEncoderDecoderTransformerInput'
                  <$> mkTransformerInput
                        padTokenId
                        batchDim
                        seqDim
                        device
                        encoderIds
      
      (SimplifiedEncoderDecoderTransformerOutput' encoderOutput paddingMask, g') <- forward model input g

      x <- SimplifiedEncoderDecoderTransformerGenerationInput 
                <$> mkTransformerInput
                      padTokenId
                      batchDim
                      (SNoName :&: SUncheckedSize 0)
                      device
                      []
                <*> pure encoderOutput
                <*> pure paddingMask

      us <- sOnes $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device (SDataType SInt64) (SShape $ batchDim :|: SNil)

      ((SimplifiedEncoderDecoderTransformerGenerationInput decoderInput _ _, _g), _us) <- flip runStateT us $ 
        greedySearch padTokenId eosTokenId model sedtOutputToInput x g'
      
      let decoderIds :: [[Int]] = fromTensor decoderInput

      traverse (Tokenizers.decode tokenizer) decoderIds

  where
    
    greedySearch padTokenId eosTokenId model zoom =
      decode (\input g -> do
        unfinishedSequences <- get
        b <- allSequencesFinished unfinishedSequences
        if b then
          pure Nothing
        else
          do
            (output, g') <- forward model input g
            input' <- (zoom . prepNext %~ greedyNextTokens padTokenId eosTokenId) output
            pure $ Just (input', g')      
      )