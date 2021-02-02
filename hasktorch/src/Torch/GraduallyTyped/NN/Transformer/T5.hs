{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.NN.Transformer.T5
  ( module Torch.GraduallyTyped.NN.Transformer.T5.Common,
    module Torch.GraduallyTyped.NN.Transformer.T5.Base,
    module Torch.GraduallyTyped.NN.Transformer.T5.Small,
    testForwardT5Small,
    testForwardT5Base,
  )
where

import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Base
import Torch.GraduallyTyped.NN.Transformer.T5.Common
import Torch.GraduallyTyped.NN.Transformer.T5.Small
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

testT5Input :: IO _
testT5Input =
  mkT5Input
    @( 'Dim ( 'Name "*") ( 'Size 2))
    @( 'Dim ( 'Name "*") ( 'Size 17))
    [ [6536, 43, 118, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1, 0, 0],
      [6536, 43, 118, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 11, 25, 1]
    ]

testT5DecoderInput :: IO _
testT5DecoderInput =
  mkT5Input
    @( 'Dim ( 'Name "*") ( 'Size 2))
    @( 'Dim ( 'Name "*") ( 'Size 4))
    [ [6536, 504, 24, 1],
      [6536, 504, 24, 1]
    ]

testForwardT5Small :: IO ()
testForwardT5Small =
  do
    input <- testT5Input
    decoderInput <- testT5DecoderInput
    model <-
      initialize
        @(T5Small 'WithLMHead ( 'Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt"
    g <- mkGenerator @( 'Device CPU) 0
    let (output, _) = forward model (input, decoderInput) g
    print output
    pure ()

testForwardT5Base :: IO ()
testForwardT5Base =
  do
    input <- testT5Input
    decoderInput <- testT5DecoderInput
    model <-
      initialize
        @(T5Base 'WithLMHead ( 'Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-base.pt"
    g <- mkGenerator @( 'Device CPU) 0
    let (output, _) = forward model (input, decoderInput) g
    print output
    pure ()
