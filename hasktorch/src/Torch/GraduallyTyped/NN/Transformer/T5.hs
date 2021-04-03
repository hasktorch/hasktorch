{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.T5
  ( module Torch.GraduallyTyped.NN.Transformer.T5.Common,
    module Torch.GraduallyTyped.NN.Transformer.T5.Base,
    module Torch.GraduallyTyped.NN.Transformer.T5.Small,
    module Torch.GraduallyTyped.NN.Transformer.T5.Large,
    module Torch.GraduallyTyped.NN.Transformer.T5.ThreeB,
    module Torch.GraduallyTyped.NN.Transformer.T5.ElevenB,
    module Torch.GraduallyTyped.NN.Transformer.T5.Generation,
    module Torch.GraduallyTyped.NN.Transformer.T5.Vocab,
    -- testForwardT5Small,
    -- testForwardT5Base,
    -- testForwardT5Large,
    -- testForwardT5ThreeB,
    -- testForwardT5ElevenB,
  )
where

import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Base
import Torch.GraduallyTyped.NN.Transformer.T5.Common
import Torch.GraduallyTyped.NN.Transformer.T5.ElevenB
import Torch.GraduallyTyped.NN.Transformer.T5.Generation
import Torch.GraduallyTyped.NN.Transformer.T5.Large
import Torch.GraduallyTyped.NN.Transformer.T5.Small
import Torch.GraduallyTyped.NN.Transformer.T5.ThreeB
import Torch.GraduallyTyped.NN.Transformer.T5.Vocab
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- testT5Input :: IO _
-- testT5Input =
--   mkT5Input
--     @('Dim ('Name "*") ('Size 2))
--     @('Dim ('Name "*") ('Size 17))
--     [ [6536, 43, 118, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1, 0, 0],
--       [6536, 43, 118, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 11, 25, 1]
--     ]

-- testT5DecoderInput :: IO _
-- testT5DecoderInput =
--   mkT5Input
--     @('Dim ('Name "*") ('Size 2))
--     @('Dim ('Name "*") ('Size 4))
--     [ [6536, 504, 24, 1],
--       [6536, 504, 24, 1]
--     ]

-- testForwardT5Small :: IO ()
-- testForwardT5Small =
--   do
--     input <- T5Input <$> testT5Input <*> testT5DecoderInput
--     model <-
--       initialize
--         @(T5SmallWithLMHead ('Device 'CPU))
--         "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt"
--     g <- mkGenerator @('Device CPU) 0
--     let (output, _) = forward model input g
--     print output

-- testForwardT5Base :: IO ()
-- testForwardT5Base =
--   do
--     input <- T5Input <$> testT5Input <*> testT5DecoderInput
--     model <-
--       initialize
--         @(T5BaseWithLMHead ('Device 'CPU))
--         "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-base.pt"
--     g <- mkGenerator @('Device CPU) 0
--     let (output, _) = forward model input g
--     print output

-- testForwardT5Large :: IO ()
-- testForwardT5Large =
--   do
--     input <- T5Input <$> testT5Input <*> testT5DecoderInput
--     model <-
--       initialize
--         @(T5LargeWithLMHead ('Device 'CPU))
--         "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-large.pt"
--     g <- mkGenerator @('Device CPU) 0
--     let (output, _) = forward model input g
--     print output

-- testForwardT5ThreeB :: IO ()
-- testForwardT5ThreeB =
--   do
--     input <- T5Input <$> testT5Input <*> testT5DecoderInput
--     model <-
--       initialize
--         @(T5ThreeBWithLMHead ('Device 'CPU))
--         "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-3b.pt"
--     g <- mkGenerator @('Device CPU) 0
--     let (output, _) = forward model input g
--     print output

-- testForwardT5ElevenB :: IO ()
-- testForwardT5ElevenB =
--   do
--     input <- T5Input <$> testT5Input <*> testT5DecoderInput
--     model <-
--       initialize
--         @(T5ElevenBWithLMHead ('Device 'CPU))
--         "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-11b.pt"
--     g <- mkGenerator @('Device CPU) 0
--     let (output, _) = forward model input g
--     print output
