{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.T5
  ( module Torch.GraduallyTyped.NN.Transformer.T5.Common,
    module Torch.GraduallyTyped.NN.Transformer.T5.Base,
    module Torch.GraduallyTyped.NN.Transformer.T5.Small,
    module Torch.GraduallyTyped.NN.Transformer.T5.Large,
    module Torch.GraduallyTyped.NN.Transformer.T5.ThreeB,
    module Torch.GraduallyTyped.NN.Transformer.T5.ElevenB,
    module Torch.GraduallyTyped.NN.Transformer.T5.Generation,
    testForwardT5Small,
  )
where

import Control.Monad.State (evalStateT)
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers
import Torch.GraduallyTyped.Device (SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (fromStateDict), stateDictFromPretrained)
import Torch.GraduallyTyped.NN.Transformer.T5.Base
import Torch.GraduallyTyped.NN.Transformer.T5.Common
import Torch.GraduallyTyped.NN.Transformer.T5.ElevenB
import Torch.GraduallyTyped.NN.Transformer.T5.Generation
import Torch.GraduallyTyped.NN.Transformer.T5.Large
import Torch.GraduallyTyped.NN.Transformer.T5.Small
import Torch.GraduallyTyped.NN.Transformer.T5.ThreeB
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (SName (..), SSize (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/t5-small-tokenizer.json"

testForwardT5Small :: IO ()
testForwardT5Small =
  do
    stateDict <- stateDictFromPretrained "/tmp/t5-small-state-dict.pt"

    let spec = t5SmallSpec SWithLMHead (SGradient SWithoutGradient) (SDevice SCPU)
    model <- flip evalStateT stateDict $ fromStateDict spec mempty

    let g = sMkGenerator (SDevice SCPU) 0

    (encoderIds, decoderIds) <- withTokenizer $ \tokenizer -> do
      encoderEncoding <- Tokenizers.encode tokenizer "translate English to German: Studies have shown that owning a dog is good for you and your dog.</s>"
      decoderEncoding <- Tokenizers.encode tokenizer "Studien haben gezeigt, dass das Besitzen eines Hundes gut für Sie und Ihren Hund ist.</s>"
      (,) <$> Tokenizers.getIDs encoderEncoding <*> Tokenizers.getIDs decoderEncoding
    let encoderSeqSize = SUncheckedSize . fromIntegral $ length encoderIds
        decoderSeqSize = SUncheckedSize . fromIntegral $ length decoderIds

    input <-
      T5Input
        <$> mkT5Input
          (SName @"*" :&: SSize @1)
          (SName @"*" :&: encoderSeqSize)
          [encoderIds]
        <*> mkT5Input
          (SName @"*" :&: SSize @1)
          (SName @"*" :&: decoderSeqSize)
          [decoderIds]

    (T5Output {..}, _) <- forward model input g

    let decoderOutput = case t5DecoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
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

--     let g = sMkGenerator (SDevice SCPU) 0

--     input <-
--       T5Input
--         <$> undefined
--         <*> undefined

--     (T5Output {..}, _) <- forward model input g

--     let decoderOutput = case t5DecoderOutput of
--           UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
--     let firstLogits = do
--           firstBatch <- take 1 decoderOutput
--           firstPositions <- take 3 firstBatch
--           take 3 firstPositions
--     let firstLogits' = undefined
--     mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits' firstLogits
