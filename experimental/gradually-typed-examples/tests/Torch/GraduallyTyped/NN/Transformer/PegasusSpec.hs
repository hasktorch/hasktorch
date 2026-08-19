{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.PegasusSpec where

import Control.Monad.State (evalStateT)
import Data.Singletons (SingKind (fromSing))
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers
import Torch.GraduallyTyped

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/pegasus-xsum-tokenizer.json"

testForwardPegasusXSum :: IO ()
testForwardPegasusXSum =
  do
    stateDict <- stateDictFromFile "/tmp/pegasus-xsum-state-dict.pt"

    let device = SDevice SCPU

    let spec = pegasusXSumSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
    model <- flip evalStateT stateDict $ fromStateDict spec mempty

    g <- sMkGenerator (SDevice SCPU) 0

    (encoderIds, decoderIds) <- withTokenizer $ \tokenizer -> do
      encoderEncoding <- Tokenizers.encode tokenizer "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.</s>"
      decoderEncoding <- Tokenizers.encode tokenizer "The Eiffel Tower, built in 1889, is one of the most famous landmarks in Paris.</s>"
      (,) <$> Tokenizers.getIDs encoderEncoding <*> Tokenizers.getIDs decoderEncoding
    let batchDim = SName @"*" :&: SSize @1
        encoderSeqSize = SUncheckedSize . fromIntegral $ length encoderIds
        encoderSeqDim = SName @"*" :&: encoderSeqSize
        decoderSeqSize = SUncheckedSize . fromIntegral $ length decoderIds
        decoderSeqDim = SName @"*" :&: decoderSeqSize

    input <-
      SimplifiedEncoderDecoderTransformerInput
        <$> mkPegasusInput batchDim encoderSeqDim device [encoderIds]
        <*> mkPegasusInput batchDim decoderSeqDim device [decoderIds]

    (SimplifiedEncoderDecoderTransformerOutput {..}, _) <- forward model input g

    encoderOutput :: [[[Float]]] <-
      fromTensor
        <$> sCheckedShape
          ( SShape $
              SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ batchDim)
                :|: encoderSeqDim
                :|: SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ pegasusXSumInputEmbedDim)
                :|: SNil
          )
          sedtEncoderOutput
    let firstEncoderHiddenStates = do
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstEncoderHiddenStates' = [0.0965, -0.0048, -0.1945, -0.0825, 0.1829, -0.1589, -0.0297, -0.0171, -0.1210]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstEncoderHiddenStates' firstEncoderHiddenStates

    decoderOutput :: [[[Float]]] <-
      fromTensor
        <$> sCheckedShape
          ( SShape $
              SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ batchDim)
                :|: SName @"*" :&: (\case SUncheckedSize size -> SUncheckedSize $ size + 1) decoderSeqSize
                :|: SName @"*" :&: SUncheckedSize (forgetIsChecked . dimSize . fromSing $ pegasusXSumVocabDim)
                :|: SNil
          )
          sedtDecoderOutput
    let firstLogits = do
          firstBatch <- take 1 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLogits' = [0.0000, 4.9619, 0.4453, 0.0000, 3.7238, 0.5208, 0.0000, 4.0774, 0.1976]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits' firstLogits
