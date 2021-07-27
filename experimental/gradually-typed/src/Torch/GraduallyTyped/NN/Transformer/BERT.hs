{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.BERT
  ( module Torch.GraduallyTyped.NN.Transformer.BERT.Common,
    module Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased,
    testForwardBERTBaseUncased,
  )
where

import Control.Monad.State (evalStateT)
import Data.Singletons.Prelude.List (SList (SNil))
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers
import Torch.GraduallyTyped.DType (SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (fromStateDict), stateDictFromPretrained)
import Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased
import Torch.GraduallyTyped.NN.Transformer.BERT.Common
import Torch.GraduallyTyped.NN.Transformer.GEncoderOnly (EncoderOnlyTransformerInput (..), EncoderOnlyTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead), mkTransformerAttentionMask)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (SName (..), SShape (..), SSize (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sArangeNaturals, sZeros)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), TensorSpec (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/bert-base-uncased-tokenizer.json"

testForwardBERTBaseUncased :: IO ()
testForwardBERTBaseUncased =
  do
    stateDict <- stateDictFromPretrained "/tmp/bert-base-uncased-state-dict.pt"

    let spec = bertBaseUnchasedSpec SWithLMHead (SGradient SWithoutGradient) (SDevice SCPU)
    GBERTModel {..} <- flip evalStateT stateDict $ fromStateDict spec mempty

    let g = sMkGenerator (SDevice SCPU) 0

    ids <- withTokenizer $ \tokenizer -> do
      encoding <- Tokenizers.encode tokenizer "[CLS] the capital of france is [MASK]. [SEP]"
      Tokenizers.getIDs encoding
    let seqSize = SUncheckedSize . fromIntegral $ length ids

    input <-
      mkBERTInput
        (SName @"*" :&: SSize @1)
        (SName @"*" :&: seqSize)
        [ids]
    let inputType =
          sZeros $
            TensorSpec
              (SGradient SWithoutGradient)
              (SLayout SDense)
              (SDevice SCPU)
              (SDataType SInt64)
              (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: seqSize :|: SNil)
        pos =
          sArangeNaturals
            (SGradient SWithoutGradient)
            (SLayout SDense)
            (SDevice SCPU)
            (SDataType SInt64)
            seqSize
        paddingMask = mkBERTPaddingMask input
    attentionMask <- mkTransformerAttentionMask bertDataType bertAttentionMaskBias paddingMask

    let eotInput = EncoderOnlyTransformerInput input inputType pos attentionMask
    (EncoderOnlyTransformerOutput {..}, _) <- forward bertModel eotInput g

    let encoderOutput' = case eoEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLMHeadLogits = do
          firstBatch <- take 1 encoderOutput'
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLMHeadLogits' = [-6.4346, -6.4063, -6.4097, -14.0119, -14.7240, -14.2120, -9.6561, -10.3125, -9.7459]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits' firstLMHeadLogits
