{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Typed.NNSpec
  ( Torch.Typed.NNSpec.spec,
  )
where

import Test.Hspec
import Torch.Typed

spec :: Spec
spec = return ()

testLinear ::
  IO
    ( HList
        '[ Parameter '( 'CPU, 0) 'Float '[5, 10],
           Parameter '( 'CPU, 0) 'Float '[5]
         ]
    )
testLinear = do
  let spec = LinearSpec @10 @5 @'Float @'( 'CPU, 0)
  model <- sample spec
  pure . flattenParameters $ model

testDropout :: IO (HList '[])
testDropout = do
  let spec = DropoutSpec 0.1
  model <- sample spec
  pure . flattenParameters $ model

testConstEmbedding :: IO (HList '[])
testConstEmbedding = do
  let spec = ConstEmbeddingSpec @'Nothing @10 @8 @'Float @'( 'CPU, 0) zeros
  model <- sample spec
  pure . flattenParameters $ model

testLearnedEmbeddingWithRandomInit :: IO (HList '[Parameter '( 'CPU, 0) 'Float '[10, 8]])
testLearnedEmbeddingWithRandomInit = do
  let spec = LearnedEmbeddingWithRandomInitSpec @'Nothing @10 @8 @'Float @'( 'CPU, 0)
  model <- sample spec
  pure . flattenParameters $ model

testLearnedEmbeddingWithCustomInit :: IO (HList '[Parameter '( 'CPU, 0) 'Float '[10, 8]])
testLearnedEmbeddingWithCustomInit = do
  let spec = LearnedEmbeddingWithCustomInitSpec @'Nothing @10 @8 @'Float @'( 'CPU, 0) zeros
  model <- sample spec
  pure . flattenParameters $ model

testConv1d ::
  IO
    ( HList
        '[ Parameter '( 'CPU, 0) 'Float '[5, 10, 3],
           Parameter '( 'CPU, 0) 'Float '[5]
         ]
    )
testConv1d = do
  let spec = Conv1dSpec @10 @5 @3 @'Float @'( 'CPU, 0)
  model <- sample spec
  pure . flattenParameters $ model

testConv2d ::
  IO
    ( HList
        '[ Parameter '( 'CPU, 0) 'Float '[5, 10, 3, 2],
           Parameter '( 'CPU, 0) 'Float '[5]
         ]
    )
testConv2d = do
  let spec = Conv2dSpec @10 @5 @3 @2 @'Float @'( 'CPU, 0)
  model <- sample spec
  pure . flattenParameters $ model

testConv3d ::
  IO
    ( HList
        '[ Parameter '( 'CPU, 0) 'Float '[5, 10, 3, 2, 1],
           Parameter '( 'CPU, 0) 'Float '[5]
         ]
    )
testConv3d = do
  let spec = Conv3dSpec @10 @5 @3 @2 @1 @'Float @'( 'CPU, 0)
  model <- sample spec
  pure . flattenParameters $ model

testLayerNorm ::
  IO
    ( HList
        '[ Parameter '( 'CPU, 0) 'Float '[5],
           Parameter '( 'CPU, 0) 'Float '[5]
         ]
    )
testLayerNorm = do
  let spec = LayerNormSpec @'[5] @'Float @'( 'CPU, 0) 0.1
  model <- sample spec
  pure . flattenParameters $ model
