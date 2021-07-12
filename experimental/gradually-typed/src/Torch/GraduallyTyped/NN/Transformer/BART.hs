{-# LANGUAGE MultiWayIf #-}
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
    -- testForwardBARTBase,
    -- testBARTInput,
    -- testBARTDecoderInput,
    testBARTAutoencoder,
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
import Data.Function

-- https://github.com/hasktorch/tokenizers/blob/master/bindings/haskell/tokenizers-haskell/src/Tokenizers.hs
-- https://github.com/hasktorch/tokenizers/blob/master/bindings/haskell/tokenizers-haskell/test/Spec.hs#L127
-- https://github.com/hasktorch/tokenizers/blob/master/nix/rust.nix

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/bart-base-tokenizer.json"

-- input text -> token0
-- input text + token0 -> token1
-- input text + token0 + token1 -> token2
-- input text + token0 + token1 + token2 -> token3
-- input text + token0 + token1 + token2 + token3 -> token4
-- input text + token0 + token1 + token2 + token3 + token4 -> token5
-- input text + token0 + token1 + token2 + token3 + token4 + token5 -> token6
-- input text + token0 + token1 + token2 + token3 + token4 + token5 + token6 -> token7
-- input text + token0 + token1 + token2 + token3 + token4 + token5 + token6 + token7 -> token8
-- input text + token0 + token1 + token2 + token3 + token4 + token5 + token6 + token7 + token8 -> token9

testBARTAutoencoder :: String -> IO String
testBARTAutoencoder prompt = do
  stateDict <- stateDictFromPretrained "/tmp/bart-base-state-dict.pt"
  model <-
    flip evalStateT stateDict $
      fromStateDict (bartBaseSpec SWithLMHead (SGradient SWithoutGradient) (SDevice SCPU)) ""
  let g = mkGenerator @('Device 'CPU) 0

  withTokenizer $ \tokenizer -> do
    specialTokens <- Tokenizers.encode tokenizer "<mask></s>"
    [maskId, eosId] <- Tokenizers.getIDs specialTokens

    promptEncoding <- Tokenizers.encode tokenizer ("<s>" ++ prompt)
    promptIds <- Tokenizers.getIDs promptEncoding
    let encoderIds = promptIds ++ [maskId, eosId]
    encoderTensor <- mkBARTInput
      (SName @"*" :&: SSize @1)
      (SName @"*" :&: SUncheckedSize (fromIntegral $ length encoderIds))
      [encoderIds]

    let maxInputSize = 512
    outputIds <- flip fix [] $ \loop completionIds -> do
      let decoderIds = promptIds ++ completionIds
      decoderTensor <- mkBARTInput
        (SName @"*" :&: SSize @1)
        (SName @"*" :&: SUncheckedSize (fromIntegral $ length decoderIds))
        [decoderIds]
      
      let input = BARTInput encoderTensor decoderTensor
      (BARTOutput {..}, _) <- forward model input g
      let decoderOutput = case bartDecoderOutput of
            UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
      let (lastId, _lastLogit) = head $ do
            firstBatch <- take 1 decoderOutput
            let lastPosition = last firstBatch
            sortBy (comparing $ Down . snd) $ zip [0 :: Int ..] $ filter (/= 50108) lastPosition
      if lastId == eosId || length decoderIds + 1 >= maxInputSize
        then pure (promptIds ++ completionIds ++ [lastId])
        else loop (completionIds ++ [lastId])
    Tokenizers.decode tokenizer outputIds
