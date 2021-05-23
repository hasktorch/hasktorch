{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.NN.Transformer.BART
  ( module Torch.GraduallyTyped.NN.Transformer.BART.Common,
    module Torch.GraduallyTyped.NN.Transformer.BART.Base,
    testForwardBARTBase,
    testBARTInput,
    testBARTDecoderInput,
  )
where

import Data.List (maximumBy, sortBy)
import Data.Ord (comparing, Down (..))
import Test.HUnit.Approx (assertApproxEqual)
import qualified Tokenizers (Tokenizer, decode, encode, getIDs, withTokenizerFromConfigFile)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.BART.Base
import Torch.GraduallyTyped.NN.Transformer.BART.Common
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import qualified Torch.Tensor as Tensor (Tensor (..), asValue)

-- https://github.com/hasktorch/tokenizers/blob/master/bindings/haskell/tokenizers-haskell/src/Tokenizers.hs
-- https://github.com/hasktorch/tokenizers/blob/master/bindings/haskell/tokenizers-haskell/test/Spec.hs#L127
-- https://github.com/hasktorch/tokenizers/blob/master/nix/rust.nix

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/bart-tokenizer.json"

testBARTInput :: IO _
testBARTInput = do
  withTokenizer $ \tokenizer -> do
    encoding <- Tokenizers.encode tokenizer "<s>Haskell: I<mask></s>"
    ids <- Tokenizers.getIDs encoding
    print ids
    mkBARTInput
      @('Dim ('Name "*") ('Size 1))
      @('Dim ('Name "*") ('Size 6))
      [ids]

testBARTDecoderInput :: IO _
testBARTDecoderInput = do
  withTokenizer $ \tokenizer -> do
    encoding <- Tokenizers.encode tokenizer "<s>Haskell: I"
    ids <- Tokenizers.getIDs encoding
    print ids
    mkBARTInput
      @('Dim ('Name "*") ('Size 1))
      @('Dim ('Name "*") ('Size 5))
      [ids]

testForwardBARTBase :: IO ()
testForwardBARTBase =
  do
    input <- BARTInput <$> testBARTInput <*> testBARTDecoderInput
    model <-
      initialize
        @(BARTBaseWithLMHead ('Device 'CPU))
        "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/bart-base.pt"
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

-- let firstLogits' = [33.9049, 6.7412, 17.0702, 7.3159, -2.1131, 17.2696, -5.6340, -5.8494, 6.4185]
-- mapM_ (uncurry (assertApproxEqual "failed approximate equality check" 0.001)) $ zip firstLogits firstLogits'
