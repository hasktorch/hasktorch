{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NN where

import           Control.Monad.State.Strict
import           Torch.HList
import           Data.Kind                    (Type)
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           GHC.Generics

import qualified Torch.NN                      as A
import qualified Torch.Autograd                as A
import qualified Torch.Tensor                  as A
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter


data LinearSpec (inputFeatures :: Nat) (outputFeatures :: Nat)
                (dtype :: D.DType)
                (device :: (D.DeviceType, Nat))
  = LinearSpec deriving (Show, Eq)

data Linear (inputFeatures :: Nat) (outputFeatures :: Nat)
            (dtype :: D.DType)
            (device :: (D.DeviceType, Nat))
 where
  Linear
    :: forall inputFeatures outputFeatures dtype device
     . { linearWeight :: Parameter device dtype '[outputFeatures, inputFeatures]
       , linearBias   :: Parameter device dtype '[outputFeatures]
       }
    -> Linear inputFeatures outputFeatures dtype device
 deriving (Show, Generic)

-- | linear
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
linear
  :: _
  => Linear _ _ _ _
  -> Tensor _ _ _
  -> Tensor _ _ _
linear Linear {..} input =
  Torch.Typed.Functional.linear' (toDependent linearWeight) (toDependent linearBias) input

instance
  ( KnownNat inputFeatures
  , KnownNat outputFeatures
  , KnownDType dtype
  , KnownDevice device
  , RandDTypeIsValid device dtype
  ) => A.Randomizable (LinearSpec inputFeatures outputFeatures dtype device)
                      (Linear     inputFeatures outputFeatures dtype device)
 where
  sample LinearSpec =
    Linear <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data DropoutSpec
 where
  DropoutSpec
    :: { dropoutProbSpec :: Double }
    -> DropoutSpec
 deriving (Show, Eq)

data Dropout where
  Dropout
    :: { dropoutProb :: Double }
    -> Dropout
 deriving (Show, Generic)

dropout
  :: forall shape dtype device
   . Dropout
  -> Bool
  -> Tensor device dtype shape
  -> IO (Tensor device dtype shape)
dropout Dropout {..} dropoutTrain =
  Torch.Typed.Functional.dropout dropoutProb dropoutTrain

instance A.Randomizable DropoutSpec Dropout where
  sample DropoutSpec {..} = return $ Dropout dropoutProbSpec 

data EmbeddingType = Constant | Learned deriving (Show, Generic)

data EmbeddingSpec (paddingIdx :: Maybe Nat)
                   (numEmbeds :: Nat)
                   (embedSize :: Nat)
                   (embeddingType :: EmbeddingType)
                   (dtype :: D.DType)
                   (device :: (D.DeviceType, Nat))
 where
  ConstEmbeddingSpec
    :: forall paddingIdx numEmbeds embedSize dtype device
     . Tensor device dtype '[numEmbeds, embedSize]
    -> EmbeddingSpec paddingIdx numEmbeds embedSize 'Constant dtype device
  LearnedEmbeddingWithRandomInitSpec
    :: forall paddingIdx numEmbeds embedSize dtype device
     . EmbeddingSpec paddingIdx numEmbeds embedSize 'Learned dtype device
  LearnedEmbeddingWithCustomInitSpec
    :: forall paddingIdx numEmbeds embedSize dtype device
     . Tensor device dtype '[numEmbeds, embedSize]
    -> EmbeddingSpec paddingIdx numEmbeds embedSize 'Learned dtype device

deriving instance Show (EmbeddingSpec paddingIdx numEmbeds embedSize embeddingType dtype device)

data Embedding (paddingIdx :: Maybe Nat)
               (numEmbeds :: Nat)
               (embedSize :: Nat)
               (embeddingType :: EmbeddingType)
               (dtype :: D.DType)
               (device :: (D.DeviceType, Nat))
 where
  ConstEmbedding
    :: forall paddingIdx numEmbeds embedSize dtype device
    --  . (PaddingIdxCheck paddingIdx numEmbeds)
     . { constEmbedWeights :: Tensor device dtype '[numEmbeds, embedSize] }
    -> Embedding paddingIdx numEmbeds embedSize 'Constant dtype device
  LearnedEmbedding
    :: forall paddingIdx numEmbeds embedSize dtype device
    --  . (PaddingIdxCheck paddingIdx numEmbeds)
     . { learnedEmbedWeights :: Parameter device dtype '[numEmbeds, embedSize] }
    -> Embedding paddingIdx numEmbeds embedSize 'Learned dtype device

deriving instance Show (Embedding paddingIdx numEmbeds embedSize embeddingType dtype device)

instance Generic (Embedding paddingIdx numEmbeds embedSize 'Constant dtype device) where
  type Rep (Embedding paddingIdx numEmbeds embedSize 'Constant dtype device) =
    Rec0 (Tensor device dtype '[numEmbeds, embedSize])

  from (ConstEmbedding {..}) = K1 constEmbedWeights
  to = ConstEmbedding . unK1

instance Generic (Embedding paddingIdx numEmbeds embedSize 'Learned dtype device) where
  type Rep (Embedding paddingIdx numEmbeds embedSize 'Learned dtype device) =
    Rec0 (Parameter device dtype '[numEmbeds, embedSize])

  from (LearnedEmbedding {..}) = K1 learnedEmbedWeights
  to = LearnedEmbedding . unK1

embed
  :: forall paddingIdx shape numEmbeds embedSize embeddingType dtype device
   . ( KnownMaybeNat paddingIdx
     , PaddingIdxCheck paddingIdx numEmbeds
     )
  => Embedding paddingIdx numEmbeds embedSize embeddingType dtype device
  -> Tensor device 'D.Int64 shape
  -> Tensor device dtype    (Reverse (embedSize ': (Reverse shape)))
embed ConstEmbedding {..} input = embedding @paddingIdx
  False
  False
  constEmbedWeights
  input
embed LearnedEmbedding {..} input = embedding @paddingIdx
  False
  False
  (toDependent learnedEmbedWeights)
  input

instance 
  ( 
  ) => A.Randomizable (EmbeddingSpec paddingIdx numEmbeds embedSize 'Constant dtype device)
                      (Embedding     paddingIdx numEmbeds embedSize 'Constant dtype device)
 where
  sample (ConstEmbeddingSpec tensor) = pure (ConstEmbedding tensor)

instance 
  ( KnownNat numEmbeds
  , KnownNat embedSize
  , KnownDType dtype
  , KnownDevice device
  , RandDTypeIsValid device dtype
  ) => A.Randomizable (EmbeddingSpec 'Nothing numEmbeds embedSize 'Learned dtype device)
                      (Embedding     'Nothing numEmbeds embedSize 'Learned dtype device)
 where
  sample LearnedEmbeddingWithRandomInitSpec = LearnedEmbedding <$> (makeIndependent =<< randn)
  sample (LearnedEmbeddingWithCustomInitSpec tensor) = LearnedEmbedding <$> (makeIndependent =<< (pure tensor))

instance
  ( paddingIdx <= numEmbeds
  , 1 <= numEmbeds - paddingIdx
  , (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds
  , KnownNat paddingIdx
  , KnownNat numEmbeds
  , KnownNat embedSize
  , KnownDType dtype
  , KnownDevice device
  , RandDTypeIsValid device dtype
  ) => A.Randomizable (EmbeddingSpec ('Just paddingIdx) numEmbeds embedSize 'Learned dtype device)
                      (Embedding     ('Just paddingIdx) numEmbeds embedSize 'Learned dtype device)
 where
  sample LearnedEmbeddingWithRandomInitSpec =
    let mask = cat @0 (  zeros @'[paddingIdx, embedSize]                 @'D.Bool @device
                      :. ones  @'[1, embedSize]                          @'D.Bool @device
                      :. zeros @'[numEmbeds - paddingIdx - 1, embedSize] @'D.Bool @device
                      :. HNil
                      )
    in  LearnedEmbedding <$> (makeIndependent =<< (maskedFill mask (0 :: Int) <$> (randn @'[numEmbeds, embedSize] @dtype @device)))
  sample (LearnedEmbeddingWithCustomInitSpec tensor) = LearnedEmbedding <$> (makeIndependent =<< (pure tensor))

data Conv1dSpec (inputChannelSize :: Nat) (outputChannelSize :: Nat)
                (kernelSize :: Nat)
                (dtype :: D.DType)
                (device :: (D.DeviceType, Nat))
  = Conv1dSpec deriving (Show, Eq)

data Conv1d (inputChannelSize :: Nat) (outputChannelSize :: Nat)
            (kernelSize :: Nat)
            (dtype :: D.DType)
            (device :: (D.DeviceType, Nat))
 where
  Conv1d
    :: forall inputChannelSize outputChannelSize kernelSize dtype device
     . { conv1dWeight :: Parameter device dtype '[outputChannelSize, inputChannelSize, kernelSize]
       , conv1dBias   :: Parameter device dtype '[outputChannelSize]
       }
    -> Conv1d inputChannelSize outputChannelSize kernelSize dtype device
 deriving (Show, Generic)

-- | conv1d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv1d
  :: forall stride padding
   . _
  => Conv1d _ _ _ _ _
  -> Tensor _ _ _
  -> Tensor _ _ _
conv1d Conv1d {..} input = Torch.Typed.Functional.conv1d @stride @padding
  (toDependent conv1dWeight)
  (toDependent conv1dBias)
  input

instance ( KnownNat inputChannelSize
         , KnownNat outputChannelSize
         , KnownNat kernelSize
         , KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (Conv1dSpec inputChannelSize outputChannelSize kernelSize dtype device)
                    (Conv1d     inputChannelSize outputChannelSize kernelSize dtype device)
 where
  sample Conv1dSpec =
    Conv1d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data Conv2dSpec (inputChannelSize :: Nat) (outputChannelSize :: Nat)
                (kernelSize0 :: Nat) (kernelSize1 :: Nat)
                (dtype :: D.DType)
                (device :: (D.DeviceType, Nat))
  = Conv2dSpec deriving (Show, Eq)

data Conv2d (inputChannelSize :: Nat) (outputChannelSize :: Nat)
            (kernelSize0 :: Nat) (kernelSize1 :: Nat)
            (dtype :: D.DType)
            (device :: (D.DeviceType, Nat))
 where
  Conv2d
    :: forall inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device
     . { conv2dWeight :: Parameter device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1]
       , conv2dBias   :: Parameter device dtype '[outputChannelSize]
       }
    -> Conv2d inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device
 deriving (Show, Generic)

-- | conv2d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv2d
  :: forall stride padding
   . _
  => Conv2d _ _ _ _ _ _
  -> Tensor _ _ _
  -> Tensor _ _ _
conv2d Conv2d {..} input = Torch.Typed.Functional.conv2d @stride @padding
  (toDependent conv2dWeight)
  (toDependent conv2dBias)
  input

instance ( KnownNat inputChannelSize
         , KnownNat outputChannelSize
         , KnownNat kernelSize0
         , KnownNat kernelSize1
         , KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable (Conv2dSpec inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device)
                    (Conv2d     inputChannelSize outputChannelSize kernelSize0 kernelSize1 dtype device)
 where
  sample Conv2dSpec =
    Conv2d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data Conv3dSpec (inputChannelSize :: Nat) (outputChannelSize :: Nat)
                (kernelSize0 :: Nat) (kernelSize1 :: Nat) (kernelSize2 :: Nat)
                (dtype :: D.DType)
                (device :: (D.DeviceType, Nat))
  = Conv3dSpec deriving (Show, Eq)

data Conv3d (inputChannelSize :: Nat) (outputChannelSize :: Nat)
            (kernelSize0 :: Nat) (kernelSize1 :: Nat) (kernelSize2 :: Nat)
            (dtype :: D.DType)
            (device :: (D.DeviceType, Nat))
 where
  Conv3d
    :: forall inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device
     . { conv3dWeight :: Parameter device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1, kernelSize2]
       , conv3dBias   :: Parameter device dtype '[outputChannelSize]
       }
    -> Conv3d inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device
 deriving (Show, Generic)

-- | conv3d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv3d
  :: forall stride padding
   . _
  => Conv3d _ _ _ _ _ _ _
  -> Tensor _ _ _
  -> Tensor _ _ _
conv3d Conv3d {..} input = Torch.Typed.Functional.conv3d @stride @padding
  (toDependent conv3dWeight)
  (toDependent conv3dBias)
  input

instance
  ( KnownNat inputChannelSize
  , KnownNat outputChannelSize
  , KnownNat kernelSize0
  , KnownNat kernelSize1
  , KnownNat kernelSize2
  , KnownDType dtype
  , KnownDevice device
  , RandDTypeIsValid device dtype
  ) => A.Randomizable (Conv3dSpec inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device)
                      (Conv3d     inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2 dtype device)
 where
  sample Conv3dSpec =
    Conv3d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data LayerNormSpec (normalizedShape :: [Nat]) (dtype :: D.DType) (device :: (D.DeviceType, Nat))
 where
  LayerNormSpec
    :: forall normalizedShape dtype device
     . { layerNormEpsSpec :: Double}
    -> LayerNormSpec normalizedShape dtype device
 deriving (Show, Eq)

data LayerNorm (normalizedShape :: [Nat]) (dtype :: D.DType) (device :: (D.DeviceType, Nat))
 where
  LayerNorm
    :: { layerNormWeight :: Parameter device dtype normalizedShape
       , layerNormBias   :: Parameter device dtype normalizedShape
       , layerNormEps    :: Double
       }
    -> LayerNorm normalizedShape dtype device
 deriving (Show, Generic)

layerNorm
  :: forall normalizedShape shape dtype device
   . ( EndsWith shape normalizedShape
     , KnownShape normalizedShape
     )
  => LayerNorm normalizedShape dtype device
  -> Tensor device dtype shape
  -> Tensor device dtype shape
layerNorm LayerNorm {..} = Torch.Typed.Functional.layerNorm @normalizedShape
  (toDependent layerNormWeight)
  (toDependent layerNormBias)
  layerNormEps

instance
  ( TensorOptions normalizedShape dtype device
  , RandDTypeIsValid device dtype
  ) => A.Randomizable (LayerNormSpec normalizedShape dtype device)
                      (LayerNorm     normalizedShape dtype device)
 where
  sample LayerNormSpec {..} =
    LayerNorm
      <$> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> pure layerNormEpsSpec
