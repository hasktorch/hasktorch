{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -v2 #-}

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
import Torch.GraduallyTyped.Device (SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (fromStateDict), stateDictFromFile)
import Torch.GraduallyTyped.NN.Transformer.BART.Base
import Torch.GraduallyTyped.NN.Transformer.BART.Common
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (SimplifiedEncoderDecoderTransformerInput (..), SimplifiedEncoderDecoderTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead))
import Torch.GraduallyTyped.Random (sMkGenerator)
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
  stateDict <- stateDictFromFile "/tmp/bart-base-state-dict.pt"

  let device = SDevice SCPU

  let spec = bartBaseSpec SWithLMHead (SGradient SWithoutGradient) device
  model <- flip evalStateT stateDict $ fromStateDict spec mempty

  let g = sMkGenerator device 0

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
        device
        [encoderIds]

    let maxInputSize = 512
    outputIds <- flip fix ([], g) $ \loop (completionIds, g') -> do
      let decoderIds = promptIds ++ completionIds
      decoderTensor <-
        mkBARTInput
          (SName @"*" :&: SSize @1)
          (SName @"*" :&: SUncheckedSize (fromIntegral $ length decoderIds))
          device
          [decoderIds]

      let input = SimplifiedEncoderDecoderTransformerInput encoderTensor decoderTensor
      (SimplifiedEncoderDecoderTransformerOutput {..}, g'') <- forward model input g'
      let decoderOutput = case sedtDecoderOutput of
            UnsafeTensor t -> Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
      let (lastId, _lastLogit) = head $ do
            firstBatch <- take 1 decoderOutput
            let lastPosition = last firstBatch
            sortBy (comparing $ Down . snd) $ zip [0 :: Int ..] $ filter (/= 50108) lastPosition
      if lastId == eosId || length decoderIds + 1 >= maxInputSize
        then pure (promptIds ++ completionIds ++ [lastId])
        else loop (completionIds ++ [lastId], g'')
    Tokenizers.decode tokenizer outputIds
