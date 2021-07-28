{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.Pegasus
  ( module Torch.GraduallyTyped.NN.Transformer.Pegasus.Common,
    module Torch.GraduallyTyped.NN.Transformer.Pegasus.XSum,
    testForwardPegasusXSum,
  )
where

import Control.Monad.State (evalStateT)
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers
import Torch.GraduallyTyped.Device (SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (fromStateDict), stateDictFromPretrained)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (SimplifiedEncoderDecoderTransformerInput (..), SimplifiedEncoderDecoderTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.Pegasus.Common
import Torch.GraduallyTyped.NN.Transformer.Pegasus.XSum
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (SName (..), SSize (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/pegasus-xsum-tokenizer.json"

testForwardPegasusXSum :: IO ()
testForwardPegasusXSum =
  do
    stateDict <- stateDictFromPretrained "/tmp/pegasus-xsum-state-dict.pt"

    let spec = pegasusXSumSpec SWithLMHead (SGradient SWithoutGradient) (SDevice SCPU)
    model <- flip evalStateT stateDict $ fromStateDict spec mempty

    let g = sMkGenerator (SDevice SCPU) 0

    (encoderIds, decoderIds) <- withTokenizer $ \tokenizer -> do
      encoderEncoding <- Tokenizers.encode tokenizer "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.</s>"
      decoderEncoding <- Tokenizers.encode tokenizer "The Eiffel Tower, built in 1889, is one of the most famous landmarks in Paris.</s>"
      (,) <$> Tokenizers.getIDs encoderEncoding <*> Tokenizers.getIDs decoderEncoding
    let encoderSeqSize = SUncheckedSize . fromIntegral $ length encoderIds
        decoderSeqSize = SUncheckedSize . fromIntegral $ length decoderIds

    input <-
      SimplifiedEncoderDecoderTransformerInput
        <$> mkPegasusInput
          (SName @"*" :&: SSize @1)
          (SName @"*" :&: encoderSeqSize)
          [encoderIds]
        <*> mkPegasusInput
          (SName @"*" :&: SSize @1)
          (SName @"*" :&: decoderSeqSize)
          [decoderIds]

    (SimplifiedEncoderDecoderTransformerOutput {..}, _) <- forward model input g

    let encoderOutput = case sedtEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstEncoderHiddenStates = do
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstEncoderHiddenStates' = [0.0965, -0.0048, -0.1945, -0.0825, 0.1829, -0.1589, -0.0297, -0.0171, -0.1210]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstEncoderHiddenStates' firstEncoderHiddenStates

    let decoderOutput = case sedtDecoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLogits = do
          firstBatch <- take 1 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLogits' = [0.0000, 4.9619, 0.4453, 0.0000, 3.7238, 0.5208, 0.0000, 4.0774, 0.1976]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits' firstLogits
