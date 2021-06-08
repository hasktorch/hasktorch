{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BERT
  ( module Torch.GraduallyTyped.NN.Transformer.BERT.Common,
    module Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased,
    testForwardBERTBaseUncased,
  )
where

import Test.HUnit.Approx (assertApproxEqual)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased
import Torch.GraduallyTyped.NN.Transformer.BERT.Common
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformerInput (..), EncoderOnlyTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.Type (mkTransformerAttentionMask)
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Creation (arangeNaturals, zeros)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

type TestBERTSeqDim = 'Dim ('Name "*") ('Size 9)

testBERTInput :: IO _
testBERTInput =
  mkBERTInput
    @('Dim ('Name "*") ('Size 1))
    @TestBERTSeqDim
    [ [101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]
    ]

testBERTInputType :: _
testBERTInputType =
  zeros
    @'WithoutGradient
    @('Layout 'Dense)
    @('Device 'CPU)
    @('DataType 'Int64)
    @('Shape '[ 'Dim ('Name "*") ('Size 1), TestBERTSeqDim])

testForwardBERTBaseUncased :: IO ()
testForwardBERTBaseUncased =
  do
    BERTModelWithLMHead model <-
      initialize
        @(BERTBaseUncasedWithLMHead ('Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/bert-base-uncased.pt"
    encoderInput <- testBERTInput
    let encoderInputType = testBERTInputType
        pos =
          arangeNaturals
            @'WithoutGradient
            @('Layout 'Dense)
            @('Device 'CPU)
            @('DataType 'Int64)
            @TestBERTSeqDim
        paddingMask = mkBERTPaddingMask encoderInput
        attentionMask = mkTransformerAttentionMask @BERTDType @BERTDataType bertAttentionMaskBias paddingMask
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
    let firstLMHeadLogits' = [-6.4346, -6.4063, -6.4097, -14.0119, -14.7240, -14.2120, -9.6561, -10.3125, -9.7459]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits firstLMHeadLogits'
