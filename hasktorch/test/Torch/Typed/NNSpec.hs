{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}

module Torch.Typed.NNSpec
  ( Torch.Typed.NNSpec.spec
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

spec :: Spec
spec = return ()

testLinear
  :: IO
       (HList
          '[Parameter '( 'D.CPU, 0) 'D.Float '[5, 10],
            Parameter '( 'D.CPU, 0) 'D.Float '[5]])
testLinear = do
  let spec = LinearSpec @10 @5 @'D.Float @'( 'D.CPU, 0)
  model <- A.sample spec
  pure . flattenParameters $ model

testDropout :: IO (HList '[])
testDropout = do
  let spec = DropoutSpec 0.1
  model <- A.sample spec
  pure . flattenParameters $ model

testConstEmbedding :: IO (HList '[])
testConstEmbedding = do
  let spec = ConstEmbeddingSpec @'Nothing @10 @8 @'D.Float @'( 'D.CPU, 0) zeros
  model <- A.sample spec
  pure . flattenParameters $ model

testLearnedEmbeddingWithRandomInit :: IO (HList '[Parameter '( 'D.CPU, 0) 'D.Float '[10, 8]])
testLearnedEmbeddingWithRandomInit = do
  let spec = LearnedEmbeddingWithRandomInitSpec @'Nothing @10 @8 @'D.Float @'( 'D.CPU, 0)
  model <- A.sample spec
  pure . flattenParameters $ model

testLearnedEmbeddingWithCustomInit :: IO (HList '[Parameter '( 'D.CPU, 0) 'D.Float '[10, 8]])
testLearnedEmbeddingWithCustomInit = do
  let spec = LearnedEmbeddingWithCustomInitSpec @'Nothing @10 @8 @'D.Float @'( 'D.CPU, 0) zeros
  model <- A.sample spec
  pure . flattenParameters $ model

testConv1d
  :: IO (HList '[ Parameter '( 'D.CPU, 0) 'D.Float '[5, 10, 3]
                , Parameter '( 'D.CPU, 0) 'D.Float '[5]
                ])
testConv1d = do
  let spec = Conv1dSpec @10 @5 @3 @'D.Float @'( 'D.CPU, 0)
  model <- A.sample spec
  pure . flattenParameters $ model

testConv2d
  :: IO (HList '[ Parameter '( 'D.CPU, 0) 'D.Float '[5, 10, 3, 2]
                , Parameter '( 'D.CPU, 0) 'D.Float '[5]
                ])
testConv2d = do
  let spec = Conv2dSpec @10 @5 @3 @2 @'D.Float @'( 'D.CPU, 0)
  model <- A.sample spec
  pure . flattenParameters $ model

testConv3d
  :: IO (HList '[ Parameter '( 'D.CPU, 0) 'D.Float '[5, 10, 3, 2, 1]
                , Parameter '( 'D.CPU, 0) 'D.Float '[5]
                ])
testConv3d = do
  let spec = Conv3dSpec @10 @5 @3 @2 @1 @'D.Float @'( 'D.CPU, 0)
  model <- A.sample spec
  pure . flattenParameters $ model

testLayerNorm
  :: IO
       (HList
          '[Parameter '( 'D.CPU, 0) 'D.Float '[5],
            Parameter '( 'D.CPU, 0) 'D.Float '[5]])
testLayerNorm = do
  let spec = LayerNormSpec @'[5] @'D.Float @'( 'D.CPU, 0) 0.1
  model <- A.sample spec
  pure . flattenParameters $ model
