{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.NN.Transformer.BERT
  ( module Torch.GraduallyTyped.NN.Transformer.BERT.Common,
    module Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased,
    -- testForwardBERTBaseUncased,
  )
where

import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased
import Torch.GraduallyTyped.NN.Transformer.BERT.Common
import Torch.GraduallyTyped.NN.Transformer.Type (mkTransformerInput)
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

testBERTInput :: IO _
testBERTInput =
  mkTransformerInput
    @('Dim ('Name "*") ('Size 2))
    @('Dim ('Name "*") ('Size 9))
    0
    [ [101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]
    ]

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