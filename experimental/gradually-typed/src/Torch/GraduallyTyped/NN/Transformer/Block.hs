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

module Torch.GraduallyTyped.NN.Transformer.Block where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork, TransformerFeedForwardNetworkSpec (..))
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention, SelfAttentionSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle)
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (TensorSpec (TensorSpec))

-- | Transformer encoder block consisting of self-attention and a feed-forward network.
--
-- TODO: Some transformers use LayerDrop, see https://arxiv.org/abs/1909.11556, during training.
-- To support this, we will need a layer wrapper that is either the identity function or the wrapped layer
-- based on a uniformly random draw from a supplied generator.
-- Complications will arise due to the gradual typing...
data
  TransformerBlock
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
  where
  TransformerBlock ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim.
    { -- | self-attention layer
      tbSelfAttention :: SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim,
      -- | feed-forward network
      tbFeedForwardNetwork :: TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim
    } ->
    TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim

data
  TransformerBlockSpec
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
  where
  TransformerBlockSpec ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim.
    STransformerStyle style ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim headDim ->
    SDim headEmbedDim ->
    SDim embedDim ->
    SDim queryEmbedDim ->
    SDim ffnDim ->
    Double ->
    Double ->
    TransformerBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim

type instance ModelSpec (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim) = TransformerBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim

instance
  ( selfAttention ~ SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim,
    HasInitialize selfAttention device selfAttention device,
    feedForwardNetwork ~ TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim,
    HasInitialize feedForwardNetwork device feedForwardNetwork device
  ) =>
  HasInitialize
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    generatorDevice
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    device
  where
  initialize (TransformerBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        selfAttention = IxStateT . initialize @selfAttention $ SelfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps
        feedForwardNetwork = IxStateT . initialize @feedForwardNetwork $ TransformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim dropoutP eps
     in runIxStateT (TransformerBlock <<$>> selfAttention <<*>> feedForwardNetwork) generator'

instance
  SingI style =>
  HasStateDict
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
  where
  fromStateDict (TransformerBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps) k =
    let selfAttentionSpec = SelfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps
        selfAttention ST5 = fromStateDict selfAttentionSpec (k <> "layer.0.")
        selfAttention SByT5 = fromStateDict selfAttentionSpec (k <> "layer.0.")
        selfAttention SBART = fromStateDict selfAttentionSpec k
        selfAttention SMBART = fromStateDict selfAttentionSpec k
        selfAttention SPegasus = fromStateDict selfAttentionSpec k
        selfAttention SBERT = fromStateDict selfAttentionSpec (k <> "attention.")
        selfAttention SRoBERTa = fromStateDict selfAttentionSpec (k <> "attention.")
        selfAttention SGPT2 = undefined
        feedForwardNetworkSpec = TransformerFeedForwardNetworkSpec style gradient device dataType queryEmbedDim ffnDim dropoutP eps
        feedForwardNetwork ST5 = fromStateDict feedForwardNetworkSpec (k <> "layer.1.")
        feedForwardNetwork SByT5 = fromStateDict feedForwardNetworkSpec (k <> "layer.1.")
        feedForwardNetwork SBART = fromStateDict feedForwardNetworkSpec k
        feedForwardNetwork SMBART = fromStateDict feedForwardNetworkSpec k
        feedForwardNetwork SPegasus = fromStateDict feedForwardNetworkSpec k
        feedForwardNetwork SBERT = fromStateDict feedForwardNetworkSpec k
        feedForwardNetwork SRoBERTa = fromStateDict feedForwardNetworkSpec k
        feedForwardNetwork SGPT2 = undefined
     in TransformerBlock
          <$> selfAttention style
          <*> feedForwardNetwork style
  toStateDict k TransformerBlock {..} =
    let selfAttention ST5 = toStateDict (k <> "layer.0.")
        selfAttention SByT5 = toStateDict (k <> "layer.0.")
        selfAttention SBART = toStateDict k
        selfAttention SMBART = toStateDict k
        selfAttention SPegasus = toStateDict k
        selfAttention SBERT = toStateDict (k <> "attention.")
        selfAttention SRoBERTa = toStateDict (k <> "attention.")
        selfAttention SGPT2 = undefined
        feedForwardNetwork ST5 = toStateDict (k <> "layer.1.")
        feedForwardNetwork SByT5 = toStateDict (k <> "layer.1.")
        feedForwardNetwork SBART = toStateDict k
        feedForwardNetwork SMBART = toStateDict k
        feedForwardNetwork SPegasus = toStateDict k
        feedForwardNetwork SBERT = toStateDict k
        feedForwardNetwork SRoBERTa = toStateDict k
        feedForwardNetwork SGPT2 = undefined
     in do
          () <- selfAttention (sing @style) tbSelfAttention
          () <- feedForwardNetwork (sing @style) tbFeedForwardNetwork
          pure ()

-- | 'HasForward' instance for 'TransformerBlock'.
--
-- @
--      ┌───────┐  ┌───────────────┐
--      │ query │  │ attentionBias │
--      └───┬───┘  └───────┬───────┘
--          │              │
--          ▼              │
--   tbSelfAttention◄──────┘
--          ▼
-- tbFeedForwardNetwork
--          │
--          ▼
--      ┌───────┐
--      │ query │
--      └───────┘
-- @
instance
  ( HasForward
      (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
      input
      generatorDevice
      selfAttentionOutput
      selfAttentionGeneratorOutputDevice,
    HasForward
      (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim)
      selfAttentionOutput
      selfAttentionGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    input
    generatorDevice
    output
    generatorOutputDevice
  where
  forward TransformerBlock {..} input =
    runIxStateT $
      ireturn input
        >>>= IxStateT . forward tbSelfAttention
        >>>= IxStateT . forward tbFeedForwardNetwork

testBlock :: IO _
testBlock = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      dropoutP = 0.0
      eps = 1e-6
  let g = sMkGenerator device 0
  (decoderBlock, g') <- initialize (TransformerBlockSpec ST5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps) g
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoderBlock (query, attentionBias) g'
  pure output
