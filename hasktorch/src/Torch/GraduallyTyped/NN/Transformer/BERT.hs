{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.NN.Transformer.BERT
  ( module Torch.GraduallyTyped.NN.Transformer.BERT.Common,
    module Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased,
    testForwardBERTBaseUncased,
  )
where

import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased
import Torch.GraduallyTyped.NN.Transformer.BERT.Common
import Torch.GraduallyTyped.NN.Transformer.Encoder (GTransformerEncoder (..), TransformerEncoder (..))
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)

testBERTInput :: IO _
testBERTInput =
  mkBERTInput
    @('Dim ('Name "*") ('Size 1))
    @('Dim ('Name "*") ('Size 9))
    [ [101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]
    ]

testForwardBERTBaseUncased :: IO ()
testForwardBERTBaseUncased =
  do
    -- input <- BERTInput <$> testBERTInput
    BERTModel model <-
      initialize
        @(BERTBaseUncased ('Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/bert-base-uncased.pt"
    let input =
          undefined ::
            Tensor
              'WithoutGradient
              ('Layout 'Dense)
              ('Device 'CPU)
              BERTDataType
              ( 'Shape
                  '[ 'Dim ('Name "*") ('Size 1),
                     'Dim ('Name "*") ('Size 9),
                     BERTBaseUncasedInputEmbedDim
                   ]
              )
        pos =
          undefined ::
            Tensor
              'WithoutGradient
              ('Layout 'Dense)
              ('Device 'CPU)
              ('DataType 'Int64)
              ( 'Shape
                  '[ 'Dim ('Name "*") ('Size 1),
                     'Dim ('Name "*") ('Size 9)
                   ]
              )
        attentionMask =
          undefined ::
            Tensor
              'WithoutGradient
              ('Layout 'Dense)
              ('Device 'CPU)
              BERTDataType
              ( 'Shape
                  '[ 'Dim ('Name "*") ('Size 1),
                     'Dim ('Name "*") ('Size 9),
                     'Dim ('Name "*") ('Size 9)
                   ]
              )
    g <- mkGenerator @('Device CPU) 0
    let (output, _) = forward model (input, pos, attentionMask) g
    print output