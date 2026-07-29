{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}

-- | A type-level record for transformer hyperparameters.
--
-- Writing a model /family/ (base\/large, E2B\/E4B, …) in the typed API means
-- threading half a dozen type-level numbers through every signature.  This
-- module packages the standard cure: promote a record of 'Nat's with
-- @DataKinds@, so a whole configuration is /one/ type parameter, variants
-- are one-line synonyms, and field access is a type family:
--
-- > type GPT2Small = 'TransformerConfig 768 12 64 3072 50257 1024
-- >
-- > block :: KnownConfig cfg
-- >       => ...
-- >       -> Tensor device dtype '[seq, ModelDim cfg]
-- >       -> Tensor device dtype '[seq, ModelDim cfg]
--
-- Nothing here is specific to this field set — a model with different
-- hyperparameters should define its own promoted record the same way; this
-- one covers the common transformer family and demonstrates the idiom.
-- Device and dtype are deliberately not fields: they vary per deployment,
-- not per architecture, and stay ordinary type parameters.
module Torch.Typed.Config
  ( TransformerConfig (..),
    ModelDim,
    Heads,
    HeadDim,
    FeedForwardDim,
    VocabSize,
    MaxSeqLen,
    KnownConfig,
    modelDim,
    heads,
    headDim,
    feedForwardDim,
    vocabSize,
    maxSeqLen,
  )
where

import GHC.TypeLits
import Torch.Typed.Auxiliary (natValI)

-- | The record, meant to be used /promoted/: a value of kind
-- @TransformerConfig@ is a full set of hyperparameters carried as one type.
data TransformerConfig = TransformerConfig
  { transformerModelDim :: Nat,
    transformerHeads :: Nat,
    transformerHeadDim :: Nat,
    transformerFeedForwardDim :: Nat,
    transformerVocabSize :: Nat,
    transformerMaxSeqLen :: Nat
  }

-- Record selectors do not promote, so each field gets a matching family.

type family ModelDim (c :: TransformerConfig) :: Nat where
  ModelDim ('TransformerConfig d _ _ _ _ _) = d

type family Heads (c :: TransformerConfig) :: Nat where
  Heads ('TransformerConfig _ h _ _ _ _) = h

type family HeadDim (c :: TransformerConfig) :: Nat where
  HeadDim ('TransformerConfig _ _ e _ _ _) = e

type family FeedForwardDim (c :: TransformerConfig) :: Nat where
  FeedForwardDim ('TransformerConfig _ _ _ f _ _) = f

type family VocabSize (c :: TransformerConfig) :: Nat where
  VocabSize ('TransformerConfig _ _ _ _ v _) = v

type family MaxSeqLen (c :: TransformerConfig) :: Nat where
  MaxSeqLen ('TransformerConfig _ _ _ _ _ l) = l

-- | Everything a model body typically needs to reify.
type KnownConfig (c :: TransformerConfig) =
  ( KnownNat (ModelDim c),
    KnownNat (Heads c),
    KnownNat (HeadDim c),
    KnownNat (FeedForwardDim c),
    KnownNat (VocabSize c),
    KnownNat (MaxSeqLen c)
  )

modelDim :: forall c. KnownNat (ModelDim c) => Int
modelDim = natValI @(ModelDim c)

heads :: forall c. KnownNat (Heads c) => Int
heads = natValI @(Heads c)

headDim :: forall c. KnownNat (HeadDim c) => Int
headDim = natValI @(HeadDim c)

feedForwardDim :: forall c. KnownNat (FeedForwardDim c) => Int
feedForwardDim = natValI @(FeedForwardDim c)

vocabSize :: forall c. KnownNat (VocabSize c) => Int
vocabSize = natValI @(VocabSize c)

maxSeqLen :: forall c. KnownNat (MaxSeqLen c) => Int
maxSeqLen = natValI @(MaxSeqLen c)
