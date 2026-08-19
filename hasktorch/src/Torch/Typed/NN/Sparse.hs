{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.NN.Sparse where

import Data.Proxy
import GHC.Generics
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import Torch.NN (HasForward (..), Randomizable (..))
import Torch.Typed.Auxiliary
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Parameter
import Torch.Typed.Tensor

data EmbeddingType = Constant | Learned deriving (Show, Generic)

data
  EmbeddingSpec
    (paddingIdx :: Maybe Nat)
    (numEmbeds :: Nat)
    (embedSize :: Nat)
    (embeddingType :: EmbeddingType)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  ConstEmbeddingSpec ::
    forall paddingIdx numEmbeds embedSize dtype device.
    Tensor device dtype '[numEmbeds, embedSize] ->
    EmbeddingSpec paddingIdx numEmbeds embedSize 'Constant dtype device
  LearnedEmbeddingWithRandomInitSpec ::
    forall paddingIdx numEmbeds embedSize dtype device.
    EmbeddingSpec
      paddingIdx
      numEmbeds
      embedSize
      'Learned
      dtype
      device
  LearnedEmbeddingWithCustomInitSpec ::
    forall paddingIdx numEmbeds embedSize dtype device.
    Tensor device dtype '[numEmbeds, embedSize] ->
    EmbeddingSpec paddingIdx numEmbeds embedSize 'Learned dtype device

deriving instance Show (EmbeddingSpec paddingIdx numEmbeds embedSize embeddingType dtype device)

data
  Embedding
    (paddingIdx :: Maybe Nat)
    (numEmbeds :: Nat)
    (embedSize :: Nat)
    (embeddingType :: EmbeddingType)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  ConstEmbedding ::
    forall paddingIdx numEmbeds embedSize dtype device.
    --  . (PaddingIdxCheck paddingIdx numEmbeds)
    {constEmbedWeights :: Tensor device dtype '[numEmbeds, embedSize]} ->
    Embedding
      paddingIdx
      numEmbeds
      embedSize
      'Constant
      dtype
      device
  LearnedEmbedding ::
    forall paddingIdx numEmbeds embedSize dtype device.
    --  . (PaddingIdxCheck paddingIdx numEmbeds)
    {learnedEmbedWeights :: Parameter device dtype '[numEmbeds, embedSize]} ->
    Embedding
      paddingIdx
      numEmbeds
      embedSize
      'Learned
      dtype
      device

deriving instance Show (Embedding paddingIdx numEmbeds embedSize embeddingType dtype device)

instance Generic (Embedding paddingIdx numEmbeds embedSize 'Constant dtype device) where
  type
    Rep (Embedding paddingIdx numEmbeds embedSize 'Constant dtype device) =
      Rec0 (Tensor device dtype '[numEmbeds, embedSize])
  from (ConstEmbedding {..}) = K1 constEmbedWeights
  to = ConstEmbedding . unK1

instance Generic (Embedding paddingIdx numEmbeds embedSize 'Learned dtype device) where
  type
    Rep (Embedding paddingIdx numEmbeds embedSize 'Learned dtype device) =
      Rec0 (Parameter device dtype '[numEmbeds, embedSize])
  from (LearnedEmbedding {..}) = K1 learnedEmbedWeights
  to = LearnedEmbedding . unK1

instance Parameterized (Embedding paddingIdx numEmbeds embedSize 'Constant dtype device)

instance Parameterized (Embedding paddingIdx numEmbeds embedSize 'Learned dtype device)

embed ::
  forall paddingIdx shape numEmbeds embedSize embeddingType dtype device shape'.
  ( KnownMaybeNat paddingIdx,
    PaddingIdxCheck paddingIdx numEmbeds,
    shape' ~ Reverse (embedSize ': (Reverse shape))
  ) =>
  Embedding paddingIdx numEmbeds embedSize embeddingType dtype device ->
  Tensor device 'D.Int64 shape ->
  Tensor device dtype shape'
embed ConstEmbedding {..} input =
  embedding @paddingIdx
    False
    False
    constEmbedWeights
    input
embed LearnedEmbedding {..} input =
  embedding @paddingIdx
    False
    False
    (toDependent learnedEmbedWeights)
    input

instance
  ( KnownMaybeNat paddingIdx,
    PaddingIdxCheck paddingIdx numEmbeds,
    shape' ~ Reverse (embedSize ': (Reverse shape))
  ) =>
  HasForward (Embedding paddingIdx numEmbeds embedSize embeddingType dtype device) (Tensor device 'D.Int64 shape) (Tensor device dtype shape')
  where
  forward = embed
  forwardStoch = (pure .) . forward

instance
  Randomizable
    (EmbeddingSpec paddingIdx numEmbeds embedSize 'Constant dtype device)
    (Embedding paddingIdx numEmbeds embedSize 'Constant dtype device)
  where
  sample (ConstEmbeddingSpec tensor) = pure (ConstEmbedding tensor)

instance
  ( KnownNat numEmbeds,
    KnownNat embedSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (EmbeddingSpec 'Nothing numEmbeds embedSize 'Learned dtype device)
    (Embedding 'Nothing numEmbeds embedSize 'Learned dtype device)
  where
  sample LearnedEmbeddingWithRandomInitSpec = LearnedEmbedding <$> (makeIndependent =<< randn)
  sample (LearnedEmbeddingWithCustomInitSpec tensor) = LearnedEmbedding <$> (makeIndependent =<< (pure tensor))

instance
  ( paddingIdx <= numEmbeds,
    1 <= numEmbeds - paddingIdx,
    (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds,
    KnownNat paddingIdx,
    KnownNat numEmbeds,
    KnownNat embedSize,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  Randomizable
    (EmbeddingSpec ('Just paddingIdx) numEmbeds embedSize 'Learned dtype device)
    (Embedding ('Just paddingIdx) numEmbeds embedSize 'Learned dtype device)
  where
  sample LearnedEmbeddingWithRandomInitSpec =
    let mask =
          cat @0
            ( zeros @'[paddingIdx, embedSize] @'D.Bool @device
                :. ones @'[1, embedSize] @'D.Bool @device
                :. zeros @'[numEmbeds - paddingIdx - 1, embedSize] @'D.Bool @device
                :. HNil
            )
     in LearnedEmbedding <$> (makeIndependent =<< (maskedFill mask (0 :: Int) <$> (randn @'[numEmbeds, embedSize] @dtype @device)))
  sample (LearnedEmbeddingWithCustomInitSpec tensor) = LearnedEmbedding <$> (makeIndependent =<< (pure tensor))
