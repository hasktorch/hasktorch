{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BERT.Common where

import Control.Monad.Catch (MonadThrow)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import Data.Singletons.TypeLits (SNat)
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformer, EncoderOnlyTransformerSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (MkTransformerPaddingMaskC, STransformerHead, TransformerHead (..), TransformerStyle (BERT), mkTransformerInput, mkTransformerPaddingMask, STransformerStyle (SBERT))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | BERT dType.
type BERTDType = 'Float

-- | BERT dType singleton.
bertDType :: SDType BERTDType
bertDType = sing @BERTDType

-- | BERT data type.
type BERTDataType = 'DataType BERTDType

-- | BERT data type singleton.
bertDataType :: SDataType BERTDataType
bertDataType = sing @BERTDataType

-- | BERT dropout rate.
-- 'dropout_rate = 0.1'
bertDropoutP :: Double
bertDropoutP = 0.1

-- | BERT positional encoding dimension.
type BERTPosEncDim = 'Dim ('Name "*") ('Size 512)

-- | BERT positional encoding dimension singleton.
bertPosEncDim :: SDim BERTPosEncDim
bertPosEncDim = sing @BERTPosEncDim

-- | BERT layer-norm epsilon.
-- 'layer_norm_epsilon = 1e-12'
bertEps :: Double
bertEps = 1e-12

-- | BERT maximum number of position embeddings.
-- 'max_position_embeddings = 512'
bertMaxPositionEmbeddings :: Int
bertMaxPositionEmbeddings = 512

-- | BERT padding token id.
-- 'pad_token_id = 0'
bertPadTokenId :: Int
bertPadTokenId = 0

-- | BERT attention mask bias
bertAttentionMaskBias :: Double
bertAttentionMaskBias = -10000

data
  GBERTModel
    (bertModel :: Type)
  where
  GBERTModel ::
    forall bertModel.
    { bertModel :: bertModel
    } ->
    GBERTModel bertModel

-- | BERT model.
newtype
  BERTModel
    (transformerHead :: TransformerHead)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (typeVocabDim :: Dim (Name Symbol) (Size Nat))
  where
  BERTModel ::
    forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim.
    GBERTModel
      (EncoderOnlyTransformer 'BERT transformerHead numLayers gradient device BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTPosEncDim vocabDim typeVocabDim) ->
    BERTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim
  deriving stock (Generic)

data
  BERTModelSpec
    (transformerHead :: TransformerHead)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (typeVocabDim :: Dim (Name Symbol) (Size Nat))
  where
  BERTModelSpec ::
    forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim.
    STransformerHead transformerHead ->
    SNat numLayers ->
    SGradient gradient ->
    SDevice device ->
    BERTModelSpec transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim

type instance ModelSpec (BERTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim) = BERTModelSpec transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim

instance
  ( SingI headDim,
    SingI headEmbedDim,
    SingI embedDim,
    SingI inputEmbedDim,
    SingI ffnDim,
    SingI vocabDim,
    SingI typeVocabDim,
    KnownNat numLayers,
    HasStateDict
      (EncoderOnlyTransformer 'BERT transformerHead numLayers gradient device BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTPosEncDim vocabDim typeVocabDim)
  ) =>
  HasStateDict
    (BERTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  where
  fromStateDict (BERTModelSpec transformerHead numLayers gradient device) k =
    let headDim = sing @headDim
        headEmbedDim = sing @headEmbedDim
        embedDim = sing @embedDim
        inputEmbedDim = sing @inputEmbedDim
        ffnDim = sing @ffnDim
        vocabDim = sing @vocabDim
        typeVocabDim = sing @typeVocabDim
     in BERTModel . GBERTModel <$> fromStateDict (EncoderOnlyTransformerSpec SBERT transformerHead numLayers gradient device bertDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim bertPosEncDim vocabDim typeVocabDim bertDropoutP bertEps) k
  toStateDict k (BERTModel GBERTModel {..}) = toStateDict k bertModel

mkBERTInput ::
  forall batchDim seqDim m output.
  ( MonadThrow m,
    SGetDim batchDim,
    SGetDim seqDim,
    'Shape '[batchDim, seqDim]
      ~ Seq
          ( 'Shape
              '[ 'Dim ('Name "*") 'UncheckedSize,
                 'Dim ('Name "*") 'UncheckedSize
               ]
              <+> 'Shape '[batchDim, seqDim]
          )
          ('Shape '[batchDim, seqDim]),
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[batchDim, seqDim])
  ) =>
  SDim batchDim ->
  SDim seqDim ->
  [[Int]] ->
  m output
mkBERTInput = mkTransformerInput bertPadTokenId

mkBERTPaddingMask ::
  forall gradient layout device dataType shape output.
  MkTransformerPaddingMaskC layout device dataType shape output =>
  Tensor gradient layout device dataType shape ->
  output
mkBERTPaddingMask = mkTransformerPaddingMask bertPadTokenId

data BERTInput input where
  BERTInput ::
    forall input.
    { bertInput :: input
    } ->
    BERTInput input

deriving stock instance
  ( Show input
  ) =>
  Show (BERTInput input)
