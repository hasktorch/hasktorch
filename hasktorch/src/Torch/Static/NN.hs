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

module Torch.Static.NN where

import Control.Monad.State.Strict
import GHC.TypeLits
import GHC.TypeLits.Extra
import GHC.Generics

import qualified Torch.NN as A
import qualified Torch.Autograd as A
import qualified Torch.Tensor as A
import qualified Torch.DType as D
import Torch.Static
import Torch.Static.Factories
import Torch.Static.Native

newtype Parameter dtype shape = Parameter A.IndependentTensor deriving (Show)

toDependent :: Parameter dtype shape -> Tensor dtype shape
toDependent (Parameter t) = UnsafeMkTensor $ A.toDependent t

makeIndependent :: Tensor dtype shape -> IO (Parameter dtype shape)
makeIndependent t = Parameter <$> A.makeIndependent (toDynamic t)

instance A.Parameterized (Parameter dtype shape) where
  flattenParameters (Parameter x) = [x]
  replaceOwnParameters _ = Parameter <$> A.nextParameter

data LinearSpec (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) = LinearSpec
  deriving (Show, Eq)

data Linear (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) where
  Linear
    :: forall dtype inputFeatures outputFeatures
     . { weight :: Parameter dtype '[inputFeatures, outputFeatures]
       , bias :: Parameter dtype '[outputFeatures]
       }
    -> Linear dtype inputFeatures outputFeatures
  deriving (Show, Generic)

linear
  :: forall dtype (inputFeatures :: Nat) (outputFeatures :: Nat) (shape :: [Nat]) (shape' :: [Nat])
   . ( CheckBroadcast (CheckMatMul
                         shape
                         '[inputFeatures, outputFeatures]
                         (ComputeMatMul
                            (ReverseImpl shape '[]) '[outputFeatures, inputFeatures]))
                      '[outputFeatures]
                      (ComputeBroadcast
                         (ReverseImpl
                            (CheckMatMul
                               shape
                               '[inputFeatures, outputFeatures]
                               (ComputeMatMul
                                  (ReverseImpl shape '[]) '[outputFeatures, inputFeatures]))
                            '[])
                         '[outputFeatures])
                    ~ shape')
  => Linear dtype inputFeatures outputFeatures
  -> Tensor dtype shape
  -> Tensor dtype shape'
linear Linear {..} input =
  add (matmul input (toDependent weight)) (toDependent bias)

instance A.Parameterized (Linear dtype inputFeatures outputFeatures)

instance (KnownDType dtype, KnownNat inputFeatures, KnownNat outputFeatures) => A.Randomizable (LinearSpec dtype inputFeatures outputFeatures) (Linear dtype inputFeatures outputFeatures) where
  sample LinearSpec =
    Linear <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

data Dropout where
  Dropout
    :: { dropoutProb :: Double
       , dropoutTrain :: Bool
       }
    -> Dropout
  deriving (Show, Generic)

dropout :: Dropout -> Tensor dtype shape -> IO (Tensor dtype shape)
dropout Dropout {..} = Torch.Static.Native.dropout dropoutProb dropoutTrain

data EmbeddingSpec (paddingIdx :: Maybe Nat) (dtype :: D.DType) (numEmbeds :: Nat) (embedDim :: Nat) = EmbeddingSpec
  deriving (Show, Eq)

data Embedding (paddingIdx :: Maybe Nat) (dtype :: D.DType) (numEmbeds :: Nat) (embedDim :: Nat) where
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
  => A.Randomizable (EmbeddingSpec 'Nothing dtype numEmbeds embedDim) (Embedding 'Nothing dtype numEmbeds embedDim)
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
  => A.Randomizable (EmbeddingSpec ('Just paddingIdx) dtype numEmbeds embedDim) (Embedding ('Just paddingIdx) dtype numEmbeds embedDim)
 where
  sample EmbeddingSpec =
    let mask = cat @0 (  zeros @'D.Bool @'[paddingIdx, embedDim]
                      :. ones  @'D.Bool @'[1, embedDim]
                      :. zeros @'D.Bool @'[numEmbeds - paddingIdx - 1, embedDim]
                      :. HNil
                      )
    in  Embedding <$> (makeIndependent =<< (maskedFill mask (0 :: Int) <$> (randn @dtype @'[numEmbeds, embedDim])))

