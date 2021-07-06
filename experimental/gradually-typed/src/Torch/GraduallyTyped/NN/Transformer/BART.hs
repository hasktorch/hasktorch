{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

{-# LANGUAGE ScopedTypeVariables #-}
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
import qualified Tokenizers (Tokenizer, decode, encode, getIDs, withTokenizerFromConfigFile)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (fromStateDict), stateDictFromPretrained)
import Torch.GraduallyTyped.NN.Transformer.BART.Base
import Torch.GraduallyTyped.NN.Transformer.BART.Common
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerHead (WithLMHead))
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (SName (..), SSize (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)
import Test.HUnit.Approx (assertApproxEqual)

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
    encoding <- Tokenizers.encode tokenizer "<s>Haskell: I<mask></s>"
    ids <- Tokenizers.getIDs encoding
    print ids
    mkBARTInput
      (SName @"*" :&: SSize @1)
      (SName @"*" :&: SSize @6)
      [ids]

testBARTDecoderInput :: IO _
testBARTDecoderInput = do
  withTokenizer $ \tokenizer -> do
    encoding <- Tokenizers.encode tokenizer "<s>Haskell: I"
    ids <- Tokenizers.getIDs encoding
    print ids
    mkBARTInput
      (SName @"*" :&: SSize @1)
      (SName @"*" :&: SSize @5)
      [ids]

testForwardBARTBase :: IO ()
testForwardBARTBase =
  do
    input <- BARTInput <$> testBARTInput <*> testBARTDecoderInput
    stateDict <- stateDictFromPretrained "/tmp/bart-base-state-dict.pt"
    model <-
      flip evalStateT stateDict $
        fromStateDict @(BARTBase 'WithLMHead ('Gradient 'WithoutGradient) ('Device 'CPU)) (SGradient SWithoutGradient, SDevice SCPU) ""
        -- fromStateDict @(BARTBase 'WithLMHead _ _) (SGradient SWithoutGradient, SDevice SCPU) ""
    g <- mkGenerator @('Device 'CPU) 0
    let (BARTOutput {..}, _) = forward model input g
    let encoderOutput = case bartEncoderOutput of
          UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
    let firstEncoderHiddenStates = do
          firstBatch <- take 1 encoderOutput
          firstPositions <- take 3 firstBatch
          take 3 firstPositions
    print firstEncoderHiddenStates
    -- let firstEncoderHiddenStates' = [-0.0324, 0.0121, -0.0036, 0.0885, 0.1154, -0.2264, 0.3947, 0.1037, 0.0503]
    -- mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstEncoderHiddenStates firstEncoderHiddenStates'
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
    let firstLogits' = [ 33.8621,   6.3225,  18.2816, 6.7655,  -1.4854,  14.1845, 0.5911,  -1.9006,   8.9273]
    mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits' firstLogits
