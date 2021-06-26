{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTa
  ( module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common,
    module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base,
    -- testForwardRoBERTaBase,
    testRoBERTaInput,
  )
where

import Data.Singletons.Prelude.List (SList (SNil))
import Test.HUnit.Approx (assertApproxEqual)
import Tokenizers (Tokenizer, addSpecialToken, encode, getIDs, getTokens, mkRobertaTokenizer)
import Torch.GraduallyTyped.DType (SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformerInput (..), EncoderOnlyTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base
import Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common
import Torch.GraduallyTyped.NN.Transformer.Type (mkTransformerAttentionMask)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (sDimSize), SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sArangeNaturals, sZeros)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (addScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

type TestRoBERTaSeqDim = 'Dim ('Name "*") ('Size 11)

testRobertaSeqDim :: SDim TestRoBERTaSeqDim
testRobertaSeqDim = SName @"*" :&: SSize @11

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
    (SName @"*" :&: SSize @1)
    testRobertaSeqDim
    [ ids
    -- [0, 133, 812, 9, 1470, 16, 646, 32804, 530, 8174, 2]
    ]

testRoBERTaInputType :: _
testRoBERTaInputType =
  sZeros
    SWithoutGradient
    (SLayout SDense)
    (SDevice SCPU)
    (SDataType SInt64)
    (SShape $ SName @"*" :&: SSize @1 :|: testRobertaSeqDim :|: SNil)

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
            sArangeNaturals
              SWithoutGradient
              (SLayout SDense)
              (SDevice SCPU)
              (SDataType SInt64)
              (sDimSize testRobertaSeqDim)
        paddingMask = mkRoBERTaPaddingMask encoderInput
    attentionMask <- mkTransformerAttentionMask robertaDataType robertaAttentionMaskBias paddingMask
    let input = EncoderOnlyTransformerInput encoderInput encoderInputType pos attentionMask
    g <- sMkGenerator (SDevice SCPU) 0
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
