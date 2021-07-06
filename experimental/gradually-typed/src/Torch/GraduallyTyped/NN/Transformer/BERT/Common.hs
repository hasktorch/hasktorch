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
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformer, EncoderOnlyTransformerWithLMHead)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerHead (..), TransformerStyle (BERT), mkTransformerInput, mkTransformerPaddingMask, MkTransformerPaddingMaskC)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

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

-- | BERT dropout probability type.
type BERTDropoutP = Float

-- | BERT dropout rate.
-- 'dropout_rate = 0.1'
bertDropoutP :: BERTDropoutP
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
      (BERTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim) ->
    BERTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim
  deriving stock (Generic)

type family
  BERTModelF
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
    (typeVocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  BERTModelF 'WithoutHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    EncoderOnlyTransformer 'BERT numLayers gradient device BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTPosEncDim vocabDim typeVocabDim BERTDropoutP
  BERTModelF 'WithMLMHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    EncoderOnlyTransformerWithLMHead 'BERT numLayers gradient device BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTPosEncDim vocabDim typeVocabDim BERTDropoutP

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
      (BERTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
      (SGradient gradient, SDevice device, SDataType BERTDataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim BERTPosEncDim, SDim vocabDim, SDim typeVocabDim, BERTDropoutP, Double)
  ) =>
  HasStateDict
    (BERTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
    (SGradient gradient, SDevice device)
  where
  fromStateDict (gradient, device) k =
    let headDim = sing @headDim
        headEmbedDim = sing @headEmbedDim
        embedDim = sing @embedDim
        inputEmbedDim = sing @inputEmbedDim
        ffnDim = sing @ffnDim
        vocabDim = sing @vocabDim
        typeVocabDim = sing @typeVocabDim
     in BERTModel . GBERTModel <$> fromStateDict (gradient, device, bertDataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, bertPosEncDim, vocabDim, typeVocabDim, bertDropoutP, bertEps) (k <> "bert.")
  toStateDict k (BERTModel GBERTModel {..}) = toStateDict (k <> "bert.") bertModel

mkBERTInput ::
  forall batchDim seqDim m output.
  ( MonadThrow m,
    KnownDim batchDim,
    KnownDim seqDim,
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