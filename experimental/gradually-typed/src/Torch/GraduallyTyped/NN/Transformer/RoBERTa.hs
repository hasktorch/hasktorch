{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTa
  ( module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common,
    module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base,
    testForwardRoBERTaBase,
  )
where

import Control.Monad.State (evalStateT)
import Data.Singletons.Prelude.List (SList (SNil))
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers
import Torch.GraduallyTyped.DType (SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (..), stateDictFromFile)
import Torch.GraduallyTyped.NN.Transformer.GEncoderOnly (EncoderOnlyTransformerInput (..), EncoderOnlyTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base
import Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead), mkTransformerAttentionMask)
import Torch.GraduallyTyped.Prelude (pattern (:|:))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (SName (..), SShape (..), SSize (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.Creation (sArangeNaturals, sZeros)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (addScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), TensorSpec (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/roberta-base-tokenizer.json"

testForwardRoBERTaBase :: IO ()
testForwardRoBERTaBase =
  do
    stateDict <- stateDictFromFile "/tmp/roberta-base-state-dict.pt"

    let device = SDevice SCPU

    let spec = robertaBaseSpec SWithLMHead (SGradient SWithoutGradient) device
    GRoBERTaModel {..} <- flip evalStateT stateDict $ fromStateDict spec mempty

    let g = sMkGenerator device 0

    ids <- withTokenizer $ \tokenizer -> do
      encoding <- Tokenizers.encode tokenizer "<s>The capital of France is [MASK].</s>"
      Tokenizers.getIDs encoding
    let seqSize = SUncheckedSize . fromIntegral $ length ids

    input <-
      mkRoBERTaInput
        (SName @"*" :&: SSize @1)
        (SName @"*" :&: seqSize)
        device
        [ids]
    let inputType =
          sZeros $
            TensorSpec
              (SGradient SWithoutGradient)
              (SLayout SDense)
              device
              (SDataType SInt64)
              (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: seqSize :|: SNil)
        pos =
          flip addScalar (2 :: Int) $
            sArangeNaturals
              (SGradient SWithoutGradient)
              (SLayout SDense)
              device
              (SDataType SInt64)
              seqSize
        paddingMask = mkRoBERTaPaddingMask input
    attentionMask <- mkTransformerAttentionMask robertaDataType robertaAttentionMaskBias paddingMask

    let eotInput = EncoderOnlyTransformerInput input inputType pos attentionMask
    (EncoderOnlyTransformerOutput {..}, _) <- forward robertaModel eotInput g

    let encoderOutput' = case eoEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLMHeadLogits = do
          firstBatch <- take 1 encoderOutput'
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    let firstLMHeadLogits' = [32.5267, -4.5318, 21.4297, 7.9570, -2.7508, 21.1128, -2.8331, -4.1595, 10.6294]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits' firstLMHeadLogits
