{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}

module Torch.Typed.NN.TransformerSpec
  ( Torch.Typed.NN.TransformerSpec.spec
  )
where

import           Test.Hspec

import           Torch.HList
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Parameter
import           Torch.Typed.NN
import           Torch.Typed.NN.Transformer

spec :: Spec
spec = return ()

testTransformerLM
  :: IO
       (HList
          '[Parameter '( 'D.CPU, 0) 'D.Float '[16, 32],
            Parameter '( 'D.CPU, 0) 'D.Float '[96, 32],
            Parameter '( 'D.CPU, 0) 'D.Float '[96],
            Parameter '( 'D.CPU, 0) 'D.Float '[32, 32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[10, 32],
            Parameter '( 'D.CPU, 0) 'D.Float '[10],
            Parameter '( 'D.CPU, 0) 'D.Float '[32, 10],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[96, 32],
            Parameter '( 'D.CPU, 0) 'D.Float '[96],
            Parameter '( 'D.CPU, 0) 'D.Float '[32, 32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[10, 32],
            Parameter '( 'D.CPU, 0) 'D.Float '[10],
            Parameter '( 'D.CPU, 0) 'D.Float '[32, 10],
            Parameter '( 'D.CPU, 0) 'D.Float '[32],
            Parameter '( 'D.CPU, 0) 'D.Float '[128, 32],
            Parameter '( 'D.CPU, 0) 'D.Float '[128]])
testTransformerLM = do
  let spec =
        TransformerLMSpec @2 @3 @10 @0 @16 @32 @128 @'D.Float @'( 'D.CPU, 0)
          (DropoutSpec 0.2)
          (TransformerLMLayerSpec
            (MultiheadAttentionSpec
              (DropoutSpec 0.2)
            )
            (DropoutSpec 0.2)
            0.001
            (TransformerLMMLPSpec
              (DropoutSpec 0.2)
              (DropoutSpec 0.2)
              (Activation Torch.Typed.Functional.relu)
              (Activation Torch.Typed.Functional.relu)
            )
          )
  model <- A.sample spec
  pure . flattenParameters $ model
