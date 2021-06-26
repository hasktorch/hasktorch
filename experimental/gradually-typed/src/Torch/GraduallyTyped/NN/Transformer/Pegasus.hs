{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.Pegasus
  ( module Torch.GraduallyTyped.NN.Transformer.Pegasus.Common,
    module Torch.GraduallyTyped.NN.Transformer.Pegasus.XSum,
    testForwardPegasusXSum,
    testPegasusInput,
    testPegasusDecoderInput,
  )
where

import Test.HUnit.Approx (assertApproxEqual)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.Pegasus.Common
import Torch.GraduallyTyped.NN.Transformer.Pegasus.XSum
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.Shape.Type (SName (..), SSize (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

testPegasusInput :: IO _
testPegasusInput =
  mkPegasusInput
    (SName @"*" :&: SSize @2)
    (SName @"*" :&: SSize @158)
    [ [139, 5998, 117, 56966, 7641, 4653, 108, 63292, 4954, 158, 3930, 108, 160, 109, 310, 3098, 130, 142, 13042, 121, 23761, 563, 108, 111, 109, 22246, 1557, 115, 3165, 107, 3096, 1217, 117, 2151, 108, 6542, 10557, 7641, 143, 27789, 4954, 158, 124, 276, 477, 107, 2348, 203, 1187, 108, 109, 37695, 6817, 23861, 109, 1741, 20551, 112, 460, 109, 22246, 729, 121, 4109, 1557, 115, 109, 278, 108, 114, 1560, 126, 886, 118, 6820, 231, 430, 109, 15528, 3671, 115, 351, 859, 672, 140, 1554, 115, 9844, 107, 168, 140, 109, 211, 1557, 112, 1111, 114, 3098, 113, 3251, 7641, 107, 5223, 112, 109, 663, 113, 114, 16071, 11676, 134, 109, 349, 113, 109, 5998, 115, 20636, 108, 126, 117, 239, 17916, 197, 109, 15528, 3671, 141, 27815, 7641, 23121, 4954, 250, 110, 64898, 48453, 108, 109, 37695, 6817, 117, 109, 453, 22246, 294, 121, 11570, 1557, 115, 2481, 244, 109, 4315, 4380, 15258, 22947, 107, 1],
      [202, 1834, 135, 15083, 1431, 4589, 112, 1928, 107, 202, 1834, 967, 2512, 13384, 4589, 206, 276, 662, 137, 1834, 112, 134, 205, 156, 564, 107, 202, 29519, 17113, 493, 220, 8353, 130, 112, 109, 385, 113, 203, 1811, 107, 1]
    ]

testPegasusDecoderInput :: IO _
testPegasusDecoderInput =
  mkPegasusInput
    (SName @"*" :&: SSize @2)
    (SName @"*" :&: SSize @22)
    [ [139, 37695, 6817, 108, 836, 115, 39290, 108, 117, 156, 113, 109, 205, 1808, 17501, 115, 3165, 107, 1],
      [202, 29519, 17113, 117, 114, 1834, 113, 114, 323, 113, 1928, 120, 137, 129, 20293, 112, 114, 323, 113, 4589, 107, 1]
    ]

testForwardPegasusXSum :: IO ()
testForwardPegasusXSum =
  do
    input <- PegasusInput <$> testPegasusInput <*> testPegasusDecoderInput
    model <-
      initialize
        @(PegasusXSumWithLMHead ('Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/pegasus-xsum.pt"
    g <- mkGenerator @('Device 'CPU) 0
    let (PegasusOutput {..}, _) = forward model input g
    let encoderOutput = case pegasusEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstEncoderHiddenStates = do
          firstBatch <- take 2 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstEncoderHiddenStates
    let firstEncoderHiddenStates' = [0.0965, -0.0048, -0.1945, -0.0825,  0.1829, -0.1589, -0.0297, -0.0171, -0.1210, -0.1453, -0.1224, 0.0941, -0.1849, -0.0484, 0.0711, 0.0219, -0.0233, 0.1485]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstEncoderHiddenStates firstEncoderHiddenStates'
    let decoderOutput = case pegasusDecoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstLogits = do
          firstBatch <- take 2 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstLogits
    let firstLogits' = [0.0000, 4.9619, 0.4453, 0.0000, 3.7238, 0.5208, 0.0000, 4.0774, 0.1976, 0.0000, 5.1009, 0.1397, 0.0000, 3.2329, 0.4058, 0.0000, 4.4593, 0.6729]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits firstLogits'
