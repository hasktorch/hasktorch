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
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common where

import Control.Monad.Catch (MonadThrow)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformer (..), EncoderOnlyTransformerWithLMHead (..))
import Torch.GraduallyTyped.NN.Transformer.Type (MkTransformerPaddingMaskC, TransformerHead (..), TransformerStyle (RoBERTa), mkTransformerInput, mkTransformerPaddingMask)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor, SGetDim)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | RoBERTa dType.
type RoBERTaDType = 'Float

-- | RoBERTa dType singleton.
robertaDType :: SDType RoBERTaDType
robertaDType = sing @RoBERTaDType

-- | RoBERTa data type.
type RoBERTaDataType = 'DataType RoBERTaDType

-- | RoBERTa data type singleton.
robertaDataType :: SDataType RoBERTaDataType
robertaDataType = sing @RoBERTaDataType

-- | RoBERTa dropout probability type.
type RoBERTaDropoutP = Float

-- | RoBERTa dropout rate.
-- 'dropout_rate = 0.1'
robertaDropoutP :: RoBERTaDropoutP
robertaDropoutP = 0.1

-- | RoBERTa positional encoding dimension.
--
-- Note the two extra dimensions.
type RoBERTaPosEncDim = 'Dim ('Name "*") ('Size 514)

-- | RoBERTa positional encoding dimension singleton.
robertaPosEncDim :: SDim RoBERTaPosEncDim
robertaPosEncDim = sing @RoBERTaPosEncDim

-- | RoBERTa layer-norm epsilon.
-- 'layer_norm_epsilon = 1e-5'
robertaEps :: Double
robertaEps = 1e-5

-- | RoBERTa maximum number of position embeddings.
-- 'max_position_embeddings = 514'
robertaMaxPositionEmbeddings :: Int
robertaMaxPositionEmbeddings = 514

-- | RoBERTa padding token id.
-- 'pad_token_id = 1'
robertaPadTokenId :: Int
robertaPadTokenId = 1

-- | RoBERTa begin-of-sentence token id.
-- 'bos_token_id = 0'
robertaBOSTokenId :: Int
robertaBOSTokenId = 0

-- | RoBERTa end-of-sentence token id.
-- 'eos_token_id = 0'
robertaEOSTokenId :: Int
robertaEOSTokenId = 2

-- | RoBERTa attention mask bias
robertaAttentionMaskBias :: Double
robertaAttentionMaskBias = -10000

data
  GRoBERTaModel
    (robertaModel :: Type)
  where
  GRoBERTaModel ::
    forall robertaModel.
    { robertaModel :: robertaModel
    } ->
    GRoBERTaModel robertaModel

-- | RoBERTa model.
newtype
  RoBERTaModel
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
  RoBERTaModel ::
    forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim.
    GRoBERTaModel
      (RoBERTaModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim) ->
    RoBERTaModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim
  deriving stock (Generic)

type family
  RoBERTaModelF
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
  RoBERTaModelF 'WithoutHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    EncoderOnlyTransformer 'RoBERTa numLayers gradient device RoBERTaDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim RoBERTaPosEncDim vocabDim typeVocabDim
  RoBERTaModelF 'WithMLMHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    EncoderOnlyTransformerWithLMHead 'RoBERTa numLayers gradient device RoBERTaDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim RoBERTaPosEncDim vocabDim typeVocabDim

-- instance
--   ( SingI headDim,
--     SingI headEmbedDim,
--     SingI embedDim,
--     SingI inputEmbedDim,
--     SingI ffnDim,
--     SingI vocabDim,
--     SingI typeVocabDim,
--     KnownNat numLayers,
--     HasStateDict
--       (RoBERTaModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
--   ) =>
--   HasStateDict
--     (RoBERTaModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
--   where
--   fromStateDict (gradient, device) k =
--     let headDim = sing @headDim
--         headEmbedDim = sing @headEmbedDim
--         embedDim = sing @embedDim
--         inputEmbedDim = sing @inputEmbedDim
--         ffnDim = sing @ffnDim
--         vocabDim = sing @vocabDim
--         typeVocabDim = sing @typeVocabDim
--      in RoBERTaModel . GRoBERTaModel <$> fromStateDict (gradient, device, robertaDataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, robertaPosEncDim, vocabDim, typeVocabDim, robertaDropoutP, robertaEps) (k <> "roberta.")
--   toStateDict k (RoBERTaModel GRoBERTaModel {..}) = toStateDict (k <> "roberta.") robertaModel

mkRoBERTaInput ::
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
mkRoBERTaInput = mkTransformerInput robertaPadTokenId

mkRoBERTaPaddingMask ::
  forall gradient layout device dataType shape output.
  MkTransformerPaddingMaskC layout device dataType shape output =>
  Tensor gradient layout device dataType shape ->
  output
mkRoBERTaPaddingMask = mkTransformerPaddingMask robertaPadTokenId

data RoBERTaInput input where
  RoBERTaInput ::
    forall input.
    { robertaInput :: input
    } ->
    RoBERTaInput input

deriving stock instance
  ( Show input
  ) =>
  Show (RoBERTaInput input)
