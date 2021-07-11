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
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention, CrossAttentionSpec (..))
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork, TransformerFeedForwardNetworkSpec (..))
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention, SelfAttentionSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (TensorSpec (TensorSpec))

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
  where
  TransformerDecoderBlock ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim.
    { -- | self-attention layer
      tdbSelfAttention :: SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim,
      -- | cross-attention layer
      tdbCrossAttention :: CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim,
      -- | feed-forward network
      tdbFeedForwardNetwork :: TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim
    } ->
    TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim

data
  TransformerDecoderBlockSpec
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
  where
  TransformerDecoderBlockSpec ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim.
    STransformerStyle style ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim headDim ->
    SDim headEmbedDim ->
    SDim embedDim ->
    SDim queryEmbedDim ->
    SDim keyEmbedDim ->
    SDim ffnDim ->
    Double ->
    Double ->
    TransformerDecoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim

type instance ModelSpec (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim) = TransformerDecoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim

instance
  ( selfAttention ~ SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim,
    HasInitialize selfAttention device selfAttention device,
    crossAttention ~ CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim,
    HasInitialize crossAttention device crossAttention device,
    feedForwardNetwork ~ TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim,
    HasInitialize feedForwardNetwork device feedForwardNetwork device
  ) =>
  HasInitialize
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    generatorDevice
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    device
  where
  initialize (TransformerDecoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        selfAttention = IxStateT . initialize @selfAttention $ SelfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps
        crossAttention = IxStateT . initialize @crossAttention $ CrossAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps
        feedForwardNetwork = IxStateT . initialize @feedForwardNetwork $ TransformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim dropoutP eps
     in runIxStateT (TransformerDecoderBlock <<$>> selfAttention <<*>> crossAttention <<*>> feedForwardNetwork) generator'

instance
  SingI style =>
  HasStateDict
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
  where
  fromStateDict (TransformerDecoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps) k =
    let selfAttentionSpec = SelfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps
        selfAttention ST5 = fromStateDict selfAttentionSpec (k <> "layer.0.")
        selfAttention SByT5 = fromStateDict selfAttentionSpec (k <> "layer.0.")
        selfAttention SBART = fromStateDict selfAttentionSpec k
        selfAttention SMBART = fromStateDict selfAttentionSpec k
        selfAttention SPegasus = fromStateDict selfAttentionSpec k
        selfAttention SBERT = undefined
        selfAttention SRoBERTa = undefined
        selfAttention SGPT2 = undefined
        crossAttentionSpec = CrossAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps
        crossAttention ST5 = fromStateDict crossAttentionSpec (k <> "layer.1.")
        crossAttention SByT5 = fromStateDict crossAttentionSpec (k <> "layer.1.")
        crossAttention SBART = fromStateDict crossAttentionSpec k
        crossAttention SMBART = fromStateDict crossAttentionSpec k
        crossAttention SPegasus = fromStateDict crossAttentionSpec k
        crossAttention SBERT = undefined
        crossAttention SRoBERTa = undefined
        crossAttention SGPT2 = undefined
        feedForwardNetworkSpec = TransformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim dropoutP eps
        feedForwardNetwork ST5 = fromStateDict feedForwardNetworkSpec (k <> "layer.2.")
        feedForwardNetwork SByT5 = fromStateDict feedForwardNetworkSpec (k <> "layer.2.")
        feedForwardNetwork SBART = fromStateDict feedForwardNetworkSpec k
        feedForwardNetwork SMBART = fromStateDict feedForwardNetworkSpec k
        feedForwardNetwork SPegasus = fromStateDict feedForwardNetworkSpec k
        feedForwardNetwork SBERT = undefined
        feedForwardNetwork SRoBERTa = undefined
        feedForwardNetwork SGPT2 = undefined
     in TransformerDecoderBlock
          <$> selfAttention style
          <*> crossAttention style
          <*> feedForwardNetwork style
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
      (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
      (query, decoderAttentionBias)
      generatorDevice
      selfAttentionOutput
      selfAttentionGeneratorOutputDevice,
    HasForward
      (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
      (selfAttentionOutput, key, crossAttentionBias)
      selfAttentionGeneratorOutputDevice
      crossAttentionOutput
      crossAttentionGeneratorOutputDevice,
    HasForward
      (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim)
      crossAttentionOutput
      crossAttentionGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward TransformerDecoderBlock {..} (query, key, decoderAttentionBias, crossAttentionBias) =
    runIxStateT $
      ireturn (query, decoderAttentionBias)
        >>>= IxStateT . forward tdbSelfAttention
        >>>= (\query' -> IxStateT . forward tdbCrossAttention $ (query', key, crossAttentionBias))
        >>>= IxStateT . forward tdbFeedForwardNetwork

testDecoderBlock :: IO _
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
      dropoutP = 0.0
      eps = 1e-6
  let g = sMkGenerator device 0
  (decoderBlock, g') <- initialize (TransformerDecoderBlockSpec ST5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps) g
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      decoderSeqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      decoderAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoderBlock (query, key, decoderAttentionBias, crossAttentionBias) g'
  pure output