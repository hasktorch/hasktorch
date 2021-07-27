{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BART
  ( module Torch.GraduallyTyped.NN.Transformer.BART.Common,
    module Torch.GraduallyTyped.NN.Transformer.BART.Base,
    testBARTAutoencoder,
  )
where

import Control.Monad.State (evalStateT)
import Data.Function (fix)
import Data.List (sortBy)
import Data.Ord (Down (..), comparing)
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

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/bart-base-tokenizer.json"

testBARTAutoencoder :: String -> IO String
testBARTAutoencoder prompt = do
  stateDict <- stateDictFromPretrained "/tmp/bart-base-state-dict.pt"

  let spec = bartBaseSpec SWithLMHead (SGradient SWithoutGradient) (SDevice SCPU)
  model <- flip evalStateT stateDict $ fromStateDict spec mempty

  let g = mkGenerator @('Device 'CPU) 0

  withTokenizer $ \tokenizer -> do
    specialTokens <- Tokenizers.encode tokenizer "<mask></s>"
    [maskId, eosId] <- Tokenizers.getIDs specialTokens

    promptEncoding <- Tokenizers.encode tokenizer ("<s>" ++ prompt)
    promptIds <- Tokenizers.getIDs promptEncoding
    let encoderIds = promptIds ++ [maskId, eosId]
    encoderTensor <-
      mkBARTInput
        (SName @"*" :&: SSize @1)
        (SName @"*" :&: SUncheckedSize (fromIntegral $ length encoderIds))
        [encoderIds]

    let maxInputSize = 512
    outputIds <- flip fix ([], g) $ \loop (completionIds, g') -> do
      let decoderIds = promptIds ++ completionIds
      decoderTensor <-
        mkBARTInput
          (SName @"*" :&: SSize @1)
          (SName @"*" :&: SUncheckedSize (fromIntegral $ length decoderIds))
          [decoderIds]

      let input = BARTInput encoderTensor decoderTensor
      (BARTOutput {..}, g'') <- forward model input g'
      let decoderOutput = case bartDecoderOutput of
            UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
      let (lastId, _lastLogit) = head $ do
            firstBatch <- take 1 decoderOutput
            let lastPosition = last firstBatch
            sortBy (comparing $ Down . snd) $ zip [0 :: Int ..] $ filter (/= 50108) lastPosition
      if lastId == eosId || length decoderIds + 1 >= maxInputSize
        then pure (promptIds ++ completionIds ++ [lastId])
        else loop (completionIds ++ [lastId], g'')
    Tokenizers.decode tokenizer outputIds
