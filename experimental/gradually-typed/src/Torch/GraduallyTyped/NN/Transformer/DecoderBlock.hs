{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.DecoderBlock where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention)
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.Random (Generator, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)

-- | Transformer decoder block consisting of self-attention,
-- cross-attention, and a feed-forward network.
data
  TransformerDecoderBlock
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerDecoderBlock ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
    { -- | self-attention layer
      tdbSelfAttention :: SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
      -- | cross-attention layer
      tdbCrossAttention :: CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
      -- | feed-forward network
      tdbFeedForwardNetwork :: TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

instance
  ( HasInitialize
      (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, dropoutP, Double)
      generator
      generator',
    HasInitialize
      (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim keyEmbedDim, dropoutP, Double)
      generator'
      generator'',
    HasInitialize
      (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP)
      (SGradient gradient, SDevice device, SDataType dataType, SDim queryEmbedDim, SDim ffnDim, dropoutP, Double)
      generator''
      generator'''
  ) =>
  HasInitialize
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim keyEmbedDim, SDim ffnDim, dropoutP, Double)
    generator
    generator'''
  where
  initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, ffnDim, dropoutP, eps) =
    let selfAttention = IxState . initialize $ (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps)
        crossAttention = IxState . initialize $ (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps)
        feedForwardNetwork = IxState . initialize $ (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps)
     in runIxState $ TransformerDecoderBlock <<$>> selfAttention <<*>> crossAttention <<*>> feedForwardNetwork

instance
  SingI style =>
  HasStateDict
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim keyEmbedDim, SDim ffnDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, ffnDim, dropoutP, eps) k =
    let selfAttention ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) (k <> "layer.0.")
        selfAttention SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) (k <> "layer.0.")
        selfAttention SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) k
        selfAttention SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) k
        selfAttention SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) k
        selfAttention SBERT = undefined
        selfAttention SRoBERTa = undefined
        selfAttention SGPT2 = undefined
        crossAttention ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps) (k <> "layer.1.")
        crossAttention SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps) (k <> "layer.1.")
        crossAttention SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps) k
        crossAttention SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps) k
        crossAttention SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps) k
        crossAttention SBERT = undefined
        crossAttention SRoBERTa = undefined
        crossAttention SGPT2 = undefined
        feedForwardNetwork ST5 = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) (k <> "layer.2.")
        feedForwardNetwork SByT5 = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) (k <> "layer.2.")
        feedForwardNetwork SBART = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k
        feedForwardNetwork SMBART = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k
        feedForwardNetwork SPegasus = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k
        feedForwardNetwork SBERT = undefined
        feedForwardNetwork SRoBERTa = undefined
        feedForwardNetwork SGPT2 = undefined
     in TransformerDecoderBlock
          <$> selfAttention (sing @style)
          <*> crossAttention (sing @style)
          <*> feedForwardNetwork (sing @style)
  toStateDict k TransformerDecoderBlock {..} =
    let selfAttention ST5 = toStateDict (k <> "layer.0.")
        selfAttention SByT5 = toStateDict (k <> "layer.0.")
        selfAttention SBART = toStateDict k
        selfAttention SMBART = toStateDict k
        selfAttention SPegasus = toStateDict k
        selfAttention SBERT = undefined
        selfAttention SRoBERTa = undefined
        selfAttention SGPT2 = undefined
        crossAttention ST5 = toStateDict (k <> "layer.1.")
        crossAttention SByT5 = toStateDict (k <> "layer.1.")
        crossAttention SBART = toStateDict k
        crossAttention SMBART = toStateDict k
        crossAttention SPegasus = toStateDict k
        crossAttention SBERT = undefined
        crossAttention SRoBERTa = undefined
        crossAttention SGPT2 = undefined
        feedForwardNetwork ST5 = toStateDict (k <> "layer.2.")
        feedForwardNetwork SByT5 = toStateDict (k <> "layer.2.")
        feedForwardNetwork SBART = toStateDict k
        feedForwardNetwork SMBART = toStateDict k
        feedForwardNetwork SPegasus = toStateDict k
        feedForwardNetwork SBERT = undefined
        feedForwardNetwork SRoBERTa = undefined
        feedForwardNetwork SGPT2 = undefined
     in do
          () <- selfAttention (sing @style) tdbSelfAttention
          () <- crossAttention (sing @style) tdbCrossAttention
          () <- feedForwardNetwork (sing @style) tdbFeedForwardNetwork
          pure ()

-- | 'HasForward' instance for 'TransformerDecoderBlock'.
--
-- @
-- ┌──────────────────────┐  ┌───────┐  ┌─────┐  ┌────────────────────┐
-- │ decoderAttentionBias │  │ query │  │ key │  │ crossAttentionBias │
-- └──────────┬───────────┘  └───┬───┘  └──┬──┘  └─────────┬──────────┘
--            │                  │         │               │
--            │                  ▼         │               │
--            └──────────►tdbSelfAttention │               │
--                               │         │               │
--                               ▼         ▼               │
--                            tdbCrossAttention◄───────────┘
--                               │
--                               ▼
--                     tdbFeedForwardNetwork
--                               │
--                               ▼
--                           ┌───────┐
--                           │ query │
--                           └───────┘
-- @
instance
  ( HasForward
      (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      (query, decoderAttentionBias)
      (Generator generatorDevice)
      selfAttentionOutput
      selfAttentionGeneratorOutput,
    HasForward
      (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      (selfAttentionOutput, key, crossAttentionBias)
      selfAttentionGeneratorOutput
      crossAttentionOutput
      crossAttentionGeneratorOutput,
    HasForward
      (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP)
      crossAttentionOutput
      crossAttentionGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward TransformerDecoderBlock {..} (query, key, decoderAttentionBias, crossAttentionBias) =
    runIxStateT $
      ireturn (query, decoderAttentionBias)
        >>>= IxStateT . forward tdbSelfAttention
        >>>= (\query' -> IxStateT . forward tdbCrossAttention $ (query', key, crossAttentionBias))
        >>>= IxStateT . forward tdbFeedForwardNetwork

testDecoderBlock = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      keyEmbedDim = queryEmbedDim
      ffnDim = SName @"*" :&: SSize @2048
      dropoutP :: Float = 0.0
      eps = 1e-6
  g <- sMkGenerator device 0
  let (decoderBlock, g') = initialize @(TransformerDecoderBlock 'T5 _ _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, ffnDim, dropoutP, eps) g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      decoderSeqDim = SName @"*" :&: SSize @13
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      decoderAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoderBlock (query, key, decoderAttentionBias, crossAttentionBias) g'
  pure output