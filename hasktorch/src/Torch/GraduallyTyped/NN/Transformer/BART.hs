{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.NN.Transformer.BART
  ( module Torch.GraduallyTyped.NN.Transformer.BART.Common,
    module Torch.GraduallyTyped.NN.Transformer.BART.Base,
    testForwardBARTBase,
  )
where

import Test.HUnit.Approx (assertApproxEqual)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.BART.Base
import Torch.GraduallyTyped.NN.Transformer.BART.Common
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

testBARTInput :: IO _
testBARTInput =
  mkBARTInput
    @('Dim ('Name "*") ('Size 2))
    @('Dim ('Name "*") ('Size 3))
    [ [1, 1, 1],
      [1, 1]
    ]

testBARTDecoderInput :: IO _
testBARTDecoderInput =
  mkBARTInput
    @('Dim ('Name "*") ('Size 2))
    @('Dim ('Name "*") ('Size 1))
    [ [1],
      [1]
    ]

testForwardBARTBase :: IO ()
testForwardBARTBase =
  do
    input <- BARTInput <$> testBARTInput <*> testBARTDecoderInput
    model <-
      initialize
        @(BARTBaseWithLMHead ('Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/bart-base.pt"
    g <- mkGenerator @('Device 'CPU) 0
    let (BARTOutput {..}, _) = forward model input g
    let encoderOutput = case bartEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstEncoderHiddenStates = do
          firstBatch <- take 2 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstEncoderHiddenStates
    let firstEncoderHiddenStates' = []
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstEncoderHiddenStates firstEncoderHiddenStates'
    let decoderOutput = case bartDecoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLogits = do
          firstBatch <- take 2 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstLogits
    let firstLogits' = []
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits firstLogits'
