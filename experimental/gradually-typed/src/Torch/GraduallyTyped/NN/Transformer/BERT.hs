{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BERT
  ( module Torch.GraduallyTyped.NN.Transformer.BERT.Common,
    module Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased,
    testForwardBERTBaseUncased,
  )
where

import Control.Monad.State (evalStateT)
import Data.Singletons.Prelude.List (SList (SNil))
import Test.HUnit.Approx (assertApproxEqual)
import Torch.GraduallyTyped.DType (SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (fromStateDict), stateDictFromPretrained)
import Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased
import Torch.GraduallyTyped.NN.Transformer.BERT.Common
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformerInput (..), EncoderOnlyTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerHead (..), mkTransformerAttentionMask)
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sArangeNaturals, sZeros)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

type TestBERTSeqDim = 'Dim ('Name "*") ('Size 9)

testBERTSeqDim :: SDim TestBERTSeqDim
testBERTSeqDim = SName @"*" :&: SSize @9

testBERTInput :: IO _
testBERTInput =
  mkBERTInput
    (SName @"*" :&: SSize @1)
    testBERTSeqDim
    [ [101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]
    ]

testBERTInputType :: _
testBERTInputType =
  sZeros
    (SGradient SWithoutGradient)
    (SLayout SDense)
    (SDevice SCPU)
    (SDataType SInt64)
    (SShape $ SName @"*" :&: SSize @1 :|: testBERTSeqDim :|: SNil)

testForwardBERTBaseUncased :: IO ()
testForwardBERTBaseUncased =
  do
    stateDict <- stateDictFromPretrained "/tmp/bert-base-uncased-state-dict.pt"
    BERTModel GBERTModel {..} <-
      flip evalStateT stateDict $
        fromStateDict @(BERTBaseUncased 'WithMLMHead _ _) (SGradient SWithoutGradient, SDevice SCPU) ""
    encoderInput <- testBERTInput
    let encoderInputType = testBERTInputType
        pos =
          sArangeNaturals
            (SGradient SWithoutGradient)
            (SLayout SDense)
            (SDevice SCPU)
            (SDataType SInt64)
            (sDimSize testBERTSeqDim)
        paddingMask = mkBERTPaddingMask encoderInput
    attentionMask <- mkTransformerAttentionMask bertDataType bertAttentionMaskBias paddingMask
    let input = EncoderOnlyTransformerInput encoderInput encoderInputType pos attentionMask
    g <- mkGenerator @('Device 'CPU) 0
    (EncoderOnlyTransformerOutput {..}, _) <- forward bertModel input g
    let encoderOutput' = case eoEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLMHeadLogits = do
          firstBatch <- take 1 encoderOutput'
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstLMHeadLogits
    let firstLMHeadLogits' = [-6.4346, -6.4063, -6.4097, -14.0119, -14.7240, -14.2120, -9.6561, -10.3125, -9.7459]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLMHeadLogits firstLMHeadLogits'
