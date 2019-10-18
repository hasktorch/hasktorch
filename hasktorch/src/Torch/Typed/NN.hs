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
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NN where

import           Control.Monad.State.Strict
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           GHC.Generics

import qualified Torch.NN                      as A
import qualified Torch.Autograd                as A
import qualified Torch.Tensor                  as A
import qualified Torch.DType                   as D
import           Torch.Typed.Factories
import           Torch.Typed.Native
import           Torch.Typed.Tensor

newtype Parameter dtype shape = Parameter A.IndependentTensor deriving (Show)

toDependent :: Parameter dtype shape -> Tensor dtype shape
toDependent (Parameter t) = UnsafeMkTensor $ A.toDependent t

makeIndependent :: Tensor dtype shape -> IO (Parameter dtype shape)
makeIndependent t = Parameter <$> A.makeIndependent (toDynamic t)

instance A.Parameterized (Parameter dtype shape) where
  flattenParameters (Parameter x) = [x]
  replaceOwnParameters _ = Parameter <$> A.nextParameter

instance A.Parameterized (HList '[]) where
  flattenParameters _ = []
  replaceOwnParameters = return

instance (A.Parameterized x, A.Parameterized (HList xs))
  => A.Parameterized (HList (x ': xs))
 where
  flattenParameters (x :. xs) = A.flattenParameters x <> A.flattenParameters xs
  replaceOwnParameters (x :. xs) = do
    x' <- A.replaceOwnParameters x
    xs' <- A.replaceOwnParameters xs
    return $ x' :. xs'

instance A.Randomizable (HList '[]) (HList '[]) where
  sample = return

instance (A.Randomizable xSpec x, A.Randomizable (HList xsSpec) (HList xs))
  => A.Randomizable (HList (xSpec ': xsSpec)) (HList (x ': xs))
 where
  sample (xSpec :. xsSpec) = do
    x <- A.sample xSpec
    xs <- A.sample xsSpec
    return $ x :. xs

data LinearSpec (dtype :: D.DType)
                (inputFeatures :: Nat) (outputFeatures :: Nat)
  = LinearSpec deriving (Show, Eq)

data Linear (dtype :: D.DType)
            (inputFeatures :: Nat) (outputFeatures :: Nat)
 where
  Linear
    :: forall dtype inputFeatures outputFeatures
     . { linearWeight :: Parameter dtype '[outputFeatures, inputFeatures]
       , linearBias   :: Parameter dtype '[outputFeatures]
       }
    -> Linear dtype inputFeatures outputFeatures
 deriving (Show, Generic)

-- | linear
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
linear
  :: _
  => Linear _ _ _
  -> Tensor _ _
  -> Tensor _ _
linear Linear {..} input =
  Torch.Typed.Native.linear' (toDependent linearWeight) (toDependent linearBias) input

instance A.Parameterized (Linear dtype inputFeatures outputFeatures)

instance ( KnownDType dtype
         , KnownNat inputFeatures
         , KnownNat outputFeatures
         )
  => A.Randomizable (LinearSpec dtype inputFeatures outputFeatures)
                    (Linear     dtype inputFeatures outputFeatures)
 where
  sample LinearSpec =
    Linear <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data DropoutSpec
 where
  DropoutSpec
    :: { dropoutProbSpec :: Double }
    -> DropoutSpec
 deriving (Show, Generic)

data Dropout where
  Dropout
    :: { dropoutProb :: Double }
    -> Dropout
 deriving (Show, Generic)

dropout :: Dropout -> Bool -> Tensor dtype shape -> IO (Tensor dtype shape)
dropout Dropout {..} dropoutTrain =
  Torch.Typed.Native.dropout dropoutProb dropoutTrain

instance A.Parameterized Dropout

instance A.Randomizable DropoutSpec Dropout where
  sample DropoutSpec {..} = return $ Dropout dropoutProbSpec 

data EmbeddingSpec (paddingIdx :: Maybe Nat)
                   (dtype :: D.DType)
                   (numEmbeds :: Nat)
                   (embedDim :: Nat)
  = EmbeddingSpec deriving (Show, Eq)

data Embedding (paddingIdx :: Maybe Nat)
               (dtype :: D.DType)
               (numEmbeds :: Nat)
               (embedDim :: Nat)
 where
  Embedding
    :: forall paddingIdx dtype numEmbeds embedDim
    --  . (PaddingIdxCheck paddingIdx numEmbeds)
     . { embedWeights :: Parameter dtype '[numEmbeds, embedDim] }
    -> Embedding paddingIdx dtype numEmbeds embedDim
 deriving (Show, Generic)

embed
  :: forall paddingIdx dtype shape numEmbeds embedDim
   . ( KnownMaybeNat paddingIdx
     , PaddingIdxCheck paddingIdx numEmbeds
     )
  => Embedding paddingIdx dtype numEmbeds embedDim
  -> Tensor 'D.Int64 shape
  -> Tensor dtype (Reverse (embedDim ': (Reverse shape)))
embed Embedding {..} input = embedding @paddingIdx
  False
  False
  (toDependent embedWeights)
  input

instance A.Parameterized (Embedding paddingIdx dtype numEmbeds embedDim)

instance ( KnownDType dtype
         , KnownNat numEmbeds
         , KnownNat embedDim
         )
  => A.Randomizable (EmbeddingSpec 'Nothing dtype numEmbeds embedDim)
                    (Embedding     'Nothing dtype numEmbeds embedDim)
 where
  sample EmbeddingSpec = Embedding <$> (makeIndependent =<< randn)

instance ( paddingIdx <= numEmbeds
         , 1 <= numEmbeds - paddingIdx
         , (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds
         , KnownNat paddingIdx
         , KnownDType dtype
         , KnownNat numEmbeds
         , KnownNat embedDim
         )
  => A.Randomizable (EmbeddingSpec ('Just paddingIdx) dtype numEmbeds embedDim)
                    (Embedding     ('Just paddingIdx) dtype numEmbeds embedDim)
 where
  sample EmbeddingSpec =
    let mask = cat @0 (  zeros @'D.Bool @'[paddingIdx, embedDim]
                      :. ones  @'D.Bool @'[1, embedDim]
                      :. zeros @'D.Bool @'[numEmbeds - paddingIdx - 1, embedDim]
                      :. HNil
                      )
    in  Embedding <$> (makeIndependent =<< (maskedFill mask (0 :: Int) <$> (randn @dtype @'[numEmbeds, embedDim])))

data Conv1dSpec (dtype :: D.DType)
                (inputChannelSize :: Nat) (outputChannelSize :: Nat)
                (kernelSize :: Nat)
  = Conv1dSpec deriving (Show, Eq)

data Conv1d (dtype :: D.DType)
            (inputChannelSize :: Nat) (outputChannelSize :: Nat)
            (kernelSize :: Nat)
 where
  Conv1d
    :: forall dtype inputChannelSize outputChannelSize kernelSize
     . { conv1dWeight :: Parameter dtype '[ outputChannelSize, inputChannelSize
                                          , kernelSize
                                          ]
       , conv1dBias   :: Parameter dtype '[outputChannelSize]
       }
    -> Conv1d dtype inputChannelSize outputChannelSize kernelSize
 deriving (Show, Generic)

-- | conv1d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv1d
  :: forall stride padding
   . _
  => Conv1d _ _ _ _
  -> Tensor _ _
  -> Tensor _ _
conv1d Conv1d {..} input = Torch.Typed.Native.conv1d @stride @padding
  (toDependent conv1dWeight)
  (toDependent conv1dBias)
  input

instance A.Parameterized (Conv1d dtype inputChannelSize outputChannelSize kernelSize)

instance ( KnownDType dtype
         , KnownNat inputChannelSize
         , KnownNat outputChannelSize
         , KnownNat kernelSize
         )
  => A.Randomizable (Conv1dSpec dtype inputChannelSize outputChannelSize kernelSize)
                    (Conv1d     dtype inputChannelSize outputChannelSize kernelSize)
 where
  sample Conv1dSpec =
    Conv1d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data Conv2dSpec (dtype :: D.DType)
                (inputChannelSize :: Nat) (outputChannelSize :: Nat)
                (kernelSize0 :: Nat) (kernelSize1 :: Nat)
  = Conv2dSpec deriving (Show, Eq)

data Conv2d (dtype :: D.DType)
            (inputChannelSize :: Nat) (outputChannelSize :: Nat)
            (kernelSize0 :: Nat) (kernelSize1 :: Nat)
 where
  Conv2d
    :: forall dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1
     . { conv2dWeight :: Parameter dtype '[ outputChannelSize, inputChannelSize
                                          , kernelSize0, kernelSize1
                                          ]
       , conv2dBias   :: Parameter dtype '[outputChannelSize]
       }
    -> Conv2d dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1
 deriving (Show, Generic)

-- | conv2d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv2d
  :: forall stride padding
   . _
  => Conv2d _ _ _ _ _
  -> Tensor _ _
  -> Tensor _ _
conv2d Conv2d {..} input = Torch.Typed.Native.conv2d @stride @padding
  (toDependent conv2dWeight)
  (toDependent conv2dBias)
  input

instance A.Parameterized (Conv2d dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1)

instance ( KnownDType dtype
         , KnownNat inputChannelSize
         , KnownNat outputChannelSize
         , KnownNat kernelSize0
         , KnownNat kernelSize1
         )
  => A.Randomizable (Conv2dSpec dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1)
                    (Conv2d     dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1)
 where
  sample Conv2dSpec =
    Conv2d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data Conv3dSpec (dtype :: D.DType)
                (inputChannelSize :: Nat) (outputChannelSize :: Nat)
                (kernelSize0 :: Nat) (kernelSize1 :: Nat) (kernelSize2 :: Nat)
  = Conv3dSpec deriving (Show, Eq)

data Conv3d (dtype :: D.DType)
            (inputChannelSize :: Nat) (outputChannelSize :: Nat)
            (kernelSize0 :: Nat) (kernelSize1 :: Nat) (kernelSize2 :: Nat)
 where
  Conv3d
    :: forall dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2
     . { conv3dWeight :: Parameter dtype '[ outputChannelSize, inputChannelSize
                                          , kernelSize0, kernelSize1, kernelSize2
                                          ]
       , conv3dBias   :: Parameter dtype '[outputChannelSize]
       }
    -> Conv3d dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2
 deriving (Show, Generic)

-- | conv3d
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv3d
  :: forall stride padding
   . _
  => Conv3d _ _ _ _ _ _
  -> Tensor _ _
  -> Tensor _ _
conv3d Conv3d {..} input = Torch.Typed.Native.conv3d @stride @padding
  (toDependent conv3dWeight)
  (toDependent conv3dBias)
  input

instance A.Parameterized (Conv3d dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2)

instance ( KnownDType dtype
         , KnownNat inputChannelSize
         , KnownNat outputChannelSize
         , KnownNat kernelSize0
         , KnownNat kernelSize1
         , KnownNat kernelSize2
         )
  => A.Randomizable (Conv3dSpec dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2)
                    (Conv3d     dtype inputChannelSize outputChannelSize kernelSize0 kernelSize1 kernelSize2)
 where
  sample Conv3dSpec =
    Conv3d <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data LayerNormSpec (dtype :: D.DType) (normalizedShape :: [Nat])
 where
  LayerNormSpec
    :: forall dtype normalizedShape
     . { layerNormEpsSpec :: Double}
    -> LayerNormSpec dtype normalizedShape
 deriving (Show, Eq)

data LayerNorm (dtype :: D.DType) (normalizedShape :: [Nat])
 where
  LayerNorm
    :: { layerNormWeight :: Parameter dtype normalizedShape
       , layerNormBias :: Parameter dtype normalizedShape
       , layerNormEps :: Double
       }
    -> LayerNorm dtype normalizedShape
 deriving (Show, Generic)

layerNorm
  :: forall normalizedShape dtype shape
   . ( EndsWith shape normalizedShape
     , KnownShape normalizedShape
     )
  => LayerNorm dtype normalizedShape
  -> Tensor dtype shape
  -> Tensor dtype shape
layerNorm LayerNorm {..} = Torch.Typed.Native.layerNorm @normalizedShape
  (toDependent layerNormWeight)
  (toDependent layerNormBias)
  layerNormEps

instance A.Parameterized (LayerNorm dtype normalizedShape) where
  flattenParameters LayerNorm {..} =
    A.flattenParameters layerNormWeight <> A.flattenParameters layerNormBias
  replaceOwnParameters LayerNorm {..} = do
    layerNormWeight <- Parameter <$> A.nextParameter
    layerNormBias   <- Parameter <$> A.nextParameter
    return $ LayerNorm { .. }

instance (TensorOptions dtype normalizedShape)
  => A.Randomizable (LayerNormSpec dtype normalizedShape)
                    (LayerNorm     dtype normalizedShape)
 where
  sample LayerNormSpec {..} =
    LayerNorm
      <$> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> pure layerNormEpsSpec
