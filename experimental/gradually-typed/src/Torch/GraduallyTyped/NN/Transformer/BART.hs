{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BART
  ( module Torch.GraduallyTyped.NN.Transformer.BART.Common,
    module Torch.GraduallyTyped.NN.Transformer.BART.Base,
    testForwardBARTBase,
    testBARTInput,
    testBARTDecoderInput,
  )
where

import Control.Monad.State (evalStateT)
import Data.List (sortBy)
import Data.Ord (Down (..), comparing)
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers (Tokenizer, decode, encode, getIDs, withTokenizerFromConfigFile)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (fromStateDict), stateDictFromPretrained)
import Torch.GraduallyTyped.NN.Transformer.BART.Base
import Torch.GraduallyTyped.NN.Transformer.BART.Common
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead))
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (SName (..), SSize (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

-- https://github.com/hasktorch/tokenizers/blob/master/bindings/haskell/tokenizers-haskell/src/Tokenizers.hs
-- https://github.com/hasktorch/tokenizers/blob/master/bindings/haskell/tokenizers-haskell/test/Spec.hs#L127
-- https://github.com/hasktorch/tokenizers/blob/master/nix/rust.nix

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/bart-base-tokenizer.json"

testBARTInput :: IO _
testBARTInput = do
  withTokenizer $ \tokenizer -> do
    -- encoding <- Tokenizers.encode tokenizer "<s>Haskell: I<mask></s>"
    encoding <- Tokenizers.encode tokenizer "<s>hello, this is a test</s>"
    ids <- Tokenizers.getIDs encoding
    print ids
    mkBARTInput
      (SName @"*" :&: SSize @1)
      -- (SName @"*" :&: SSize @6)
      (SName @"*" :&: SSize @8)
      [ids]

testBARTDecoderInput :: IO _
testBARTDecoderInput = do
  withTokenizer $ \tokenizer -> do
    -- encoding <- Tokenizers.encode tokenizer "<s>Haskell: I"
    encoding <- Tokenizers.encode tokenizer "<s>hello, this is a test</s>"
    ids <- Tokenizers.getIDs encoding
    print ids
    mkBARTInput
      (SName @"*" :&: SSize @1)
      -- (SName @"*" :&: SSize @5)
      (SName @"*" :&: SSize @8)
      [ids]

testForwardBARTBase :: IO ()
testForwardBARTBase =
  do
    input <- BARTInput <$> testBARTInput <*> testBARTDecoderInput
    stateDict <- stateDictFromPretrained "/tmp/bart-base-state-dict.pt"
    model <-
      flip evalStateT stateDict $
        fromStateDict (bartBaseSpec SWithLMHead (SGradient SWithoutGradient) (SDevice SCPU)) ""
    let g = mkGenerator @('Device 'CPU) 0
    (BARTOutput {..}, _) <- forward model input g
    let encoderOutput = case bartEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstEncoderHiddenStates = do
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstEncoderHiddenStates
    let firstEncoderHiddenStates' = [-0.0323,  0.0127, -0.0035, 0.0310,  0.0557,  0.1267, -0.2276, -0.0942,  0.0046]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstEncoderHiddenStates firstEncoderHiddenStates'
    let decoderOutput = case bartDecoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let lastLogits = do
          firstBatch <- take 1 decoderOutput
          let lastPosition = last firstBatch
          fmap fst . sortBy (comparing $ Down . snd) $ zip [0 :: Int ..] lastPosition
    withTokenizer $ \tokenizer -> do
      stuff <-
        traverse
          ( \case
              50108 -> pure "boom!"
              idx -> Tokenizers.decode tokenizer . pure $ idx
          )
          (take 10 lastLogits)
      print stuff
    let firstLogits = do
          firstBatch <- take 1 decoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstLogits
    let firstLogits' = [33.5031,  6.3094, 16.0348, 3.0584, -2.2831, 10.3209, -4.2086, -4.3536,  0.9580]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits' firstLogits