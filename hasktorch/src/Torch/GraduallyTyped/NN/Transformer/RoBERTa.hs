{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTa
  ( module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common,
    module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base,
    testRoBERTaInput,
    testRoBERTaInputType,
    testForwardRoBERTaBase,
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

type TestRoBERTaSeqDim = 'Dim ('Name "*") ('Size 11)

testRoBERTaInput :: IO _
testRoBERTaInput =
  mkRoBERTaInput
    @('Dim ('Name "*") ('Size 1))
    @TestRoBERTaSeqDim
    [ [0, 133, 812, 9, 1470, 16, 646, 32804, 530, 8174, 2]
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
    RoBERTaModel model <-
      initialize
        @(RoBERTaBase ('Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/roberta-base.pt"
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
    let encoderOutput' = case encoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstHiddenStates = do
          firstBatch <- take 1 encoderOutput'
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstHiddenStates
    let firstHiddenStates' = [-0.0552, 0.0930, -0.0055, 0.0256, 0.2166, 0.1687, -0.0669, 0.2208, 0.2225]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstHiddenStates firstHiddenStates'
