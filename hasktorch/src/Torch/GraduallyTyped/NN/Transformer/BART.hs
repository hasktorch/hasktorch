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
    @('Dim ('Name "*") ('Size 1))
    @('Dim ('Name "*") ('Size 7))
    [ [0, 31414, 127, 766, 16, 50264, 2]
    ]

testBARTDecoderInput :: IO _
testBARTDecoderInput =
  mkBARTInput
    @('Dim ('Name "*") ('Size 1))
    @('Dim ('Name "*") ('Size 7))
    [ [0, 31414, 127, 766, 16, 1573, 2]
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
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstEncoderHiddenStates
    let firstEncoderHiddenStates' = [-0.0324, 0.0121, -0.0036, 0.0885, 0.1154, -0.2264, 0.3947, 0.1037, 0.0503]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstEncoderHiddenStates firstEncoderHiddenStates'
    let decoderOutput = case bartDecoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLogits = do
          firstBatch <- take 1 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstLogits
    let firstLogits' = [33.9049, 6.7412, 17.0702, 7.3159, -2.1131, 17.2696, -5.6340, -5.8494, 6.4185]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits firstLogits'
