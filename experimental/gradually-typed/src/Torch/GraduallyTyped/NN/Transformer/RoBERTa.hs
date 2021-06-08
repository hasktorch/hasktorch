{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTa
  ( module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common,
    module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base,
    testForwardRoBERTaBase,
    testRoBERTaInput
  )
where

import Test.HUnit.Approx (assertApproxEqual)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformerInput (..), EncoderOnlyTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base
import Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common
import Torch.GraduallyTyped.NN.Transformer.Type (mkTransformerAttentionMask)
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Creation (arangeNaturals, zeros)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (addScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)
import Tokenizers (addSpecialToken, mkRobertaTokenizer, Tokenizer, encode, getIDs, getTokens)

type TestRoBERTaSeqDim = 'Dim ('Name "*") ('Size 11)

testTokenizer :: IO Tokenizer
testTokenizer =
  mkRobertaTokenizer
    "/Users/tscholak/Projects/thirdParty/tokenizers/bindings/haskell/tokenizers-haskell/roberta-base-vocab.json"
    "/Users/tscholak/Projects/thirdParty/tokenizers/bindings/haskell/tokenizers-haskell/roberta-base-merges.txt"

testRoBERTaInput :: IO _
testRoBERTaInput = do
  tokenizer <- testTokenizer
  mapM_ (addSpecialToken tokenizer) ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
  encoded <- encode tokenizer "<s>Hello my name is<mask></s>"
  ids <- getIDs encoded
  print =<< getTokens encoded
  mkRoBERTaInput
    @('Dim ('Name "*") ('Size 1))
    @TestRoBERTaSeqDim
    [ ids
      -- [0, 133, 812, 9, 1470, 16, 646, 32804, 530, 8174, 2]
    ]

testRoBERTaInputType :: _
testRoBERTaInputType =
  zeros
    @'WithoutGradient
    @('Layout 'Dense)
    @('Device 'CPU)
    @('DataType 'Int64)
    @('Shape '[ 'Dim ('Name "*") ('Size 1), TestRoBERTaSeqDim])

testForwardRoBERTaBase :: IO ()
testForwardRoBERTaBase =
  do
    RoBERTaModelWithLMHead model <-
      initialize
        @(RoBERTaBaseWithLMHead ('Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/bert-base-uncased.pt"
    encoderInput <- testRoBERTaInput
    let encoderInputType = testRoBERTaInputType
        pos =
          flip addScalar (2 :: Int) $
            arangeNaturals
              @'WithoutGradient
              @('Layout 'Dense)
              @('Device 'CPU)
              @('DataType 'Int64)
              @TestRoBERTaSeqDim
        paddingMask = mkRoBERTaPaddingMask encoderInput
        attentionMask = mkTransformerAttentionMask @RoBERTaDType @RoBERTaDataType robertaAttentionMaskBias paddingMask
        input = EncoderOnlyTransformerInput encoderInput encoderInputType pos attentionMask
    g <- mkGenerator @('Device 'CPU) 0
    let (EncoderOnlyTransformerOutput {..}, _) = forward model input g
    let encoderOutput' = case eoEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLMHeadLogits = do
          firstBatch <- take 1 encoderOutput'
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstLMHeadLogits
    let firstLMHeadLogits' = [32.5267, -4.5318, 21.4297, 7.9570, -2.7508, 21.1128, -2.8331, -4.1595, 10.6294]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits firstLMHeadLogits'
