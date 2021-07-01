{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.T5
  ( module Torch.GraduallyTyped.NN.Transformer.T5.Common,
    module Torch.GraduallyTyped.NN.Transformer.T5.Base,
    module Torch.GraduallyTyped.NN.Transformer.T5.Small,
    module Torch.GraduallyTyped.NN.Transformer.T5.Large,
    module Torch.GraduallyTyped.NN.Transformer.T5.ThreeB,
    module Torch.GraduallyTyped.NN.Transformer.T5.ElevenB,
    module Torch.GraduallyTyped.NN.Transformer.T5.Generation,
    testForwardT5Small,
    testForwardByT5Small,
    -- testForwardT5Base,
    -- testForwardT5Large,
    -- testForwardT5ThreeB,
    -- testForwardT5ElevenB,
  )
where

import Data.Singletons.Prelude.List (SList (..))
import Test.HUnit.Approx (assertApproxEqual)
import Torch.GraduallyTyped.DType (SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Base
import Torch.GraduallyTyped.NN.Transformer.T5.Common
import Torch.GraduallyTyped.NN.Transformer.T5.ElevenB
import Torch.GraduallyTyped.NN.Transformer.T5.Generation
import Torch.GraduallyTyped.NN.Transformer.T5.Large
import Torch.GraduallyTyped.NN.Transformer.T5.Small
import Torch.GraduallyTyped.NN.Transformer.T5.ThreeB
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (SName (..), SShape (..), SSize (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

testT5BatchDim = SName @"*" :&: SSize @2

testT5InputSeqDim = SName @"*" :&: SSize @51

testT5DecoderInputSeqDim = SName @"*" :&: SSize @42

testT5Input :: IO _
testT5Input =
  mkT5Input
    testT5BatchDim
    testT5InputSeqDim
    [ [13959, 1566, 12, 2968, 10, 6536, 43, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 11, 39, 1782, 5, 1],
      [21603, 10, 9900, 1036, 6, 213, 3, 9, 825, 19, 166, 554, 18, 17, 10761, 30, 3, 9, 331, 18, 3723, 2491, 274, 271, 1399, 18, 17, 444, 26, 30, 3, 9, 26804, 2491, 6, 65, 13999, 38, 3, 9, 2021, 3317, 16, 793, 1612, 3026, 41, 567, 6892, 137, 1]
    ]

testT5DecoderInput :: IO _
testT5DecoderInput =
  mkT5Input
    testT5BatchDim
    testT5DecoderInputSeqDim
    [ [16258, 745, 22985, 6, 602, 211, 493, 10252, 35, 266, 7, 14216, 7, 1806, 218, 292, 64, 1197, 29, 14587, 229, 5, 1],
      [2025, 1036, 19, 3, 9, 2021, 3317, 16, 793, 1612, 3026, 5, 3, 9, 825, 19, 166, 554, 18, 17, 10761, 30, 3, 9, 331, 18, 3723, 2491, 274, 271, 1399, 18, 17, 444, 26, 30, 3, 9, 26804, 2491, 5, 1]
    ]

testForwardT5Small :: IO ()
testForwardT5Small =
  do
    input <- T5Input <$> testT5Input <*> testT5DecoderInput
    model <-
      initialize
        @(T5SmallWithLMHead ('Device 'CPU))
        "/tmp/t5-small-state-dict.pt"
    g <- mkGenerator @('Device 'CPU) 0
    let (T5Output {..}, _) = forward model input g
    let decoderOutput = case t5DecoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLogits = do
          firstBatch <- take 2 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstLogits
    let firstLogits' =
          [-19.3826, -10.5635, -11.4550, -26.4326, -15.4450, -14.5276, -28.7175, -14.7651, -18.2521, -14.1124, -7.4893, -12.4156, -27.9005, -11.5861, -15.9638, -24.8472, -9.6344, -12.3494]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits firstLogits'

testForwardByT5Small :: IO ()
testForwardByT5Small =
  do
    let sOnes' = sOnes SWithoutGradient (SLayout SDense) (SDevice SCPU) (SDataType SInt64)
        byT5Input = sOnes' (SShape $ testT5BatchDim :|: testT5InputSeqDim :|: SNil)
        byT5DecoderInput = sOnes' (SShape $ testT5BatchDim :|: testT5DecoderInputSeqDim :|: SNil)
        input = T5Input byT5Input byT5DecoderInput
    model <-
      initialize
        @(ByT5SmallWithLMHead ('Device 'CPU))
        "/tmp/byt5-small-state-dict.pt"
    g <- mkGenerator @('Device 'CPU) 0
    let (T5Output {..}, _) = forward model input g
    let decoderOutput = case t5DecoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLogits = do
          firstBatch <- take 2 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstLogits
    let firstLogits' =
          [-19.3826, -10.5635, -11.4550, -26.4326, -15.4450, -14.5276, -28.7175, -14.7651, -18.2521, -14.1124, -7.4893, -12.4156, -27.9005, -11.5861, -15.9638, -24.8472, -9.6344, -12.3494]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits firstLogits'

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