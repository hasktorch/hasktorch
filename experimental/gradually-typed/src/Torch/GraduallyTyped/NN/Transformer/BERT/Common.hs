{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BERT.Common where

import Control.Monad.Reader (ReaderT (runReaderT))
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import GHC.TypeNats (type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformer, EncoderOnlyTransformerWithLMHead)
import Torch.GraduallyTyped.NN.Transformer.Type (TensorDict, TransformerStyle (BERT), mkTransformerInput, mkTransformerPaddingMask, tensorDictFromPretrained)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | BERT dType.
type BERTDType = 'Float

-- | BERT dType.
bertDType :: SDType BERTDType
bertDType = SFloat

-- | BERT data type.
type BERTDataType = 'DataType BERTDType

-- | BERT data type.
bertDataType :: SDataType BERTDataType
bertDataType = SDataType bertDType

-- | BERT dropout probability type.
type BERTDropoutP = Float

-- | BERT dropout rate.
-- 'dropout_rate = 0.1'
bertDropoutP :: BERTDropoutP
bertDropoutP = 0.1

-- | BERT positional encoding dimension.
type BERTPosEncDim = 'Dim ('Name "*") ('Size 512)

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

-- | BERT model.
newtype
  BERTModel
    (numLayers :: Nat)
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
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim.
    BERTModelEncoderF BERTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim ->
    BERTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim
  deriving stock (Generic)

-- | BERT model with language modelling head.
newtype
  BERTModelWithLMHead
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (typeVocabDim :: Dim (Name Symbol) (Size Nat))
  where
  BERTModelWithLMHead ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim.
    BERTModelEncoderF BERTModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim ->
    BERTModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim
  deriving stock (Generic)

type family
  BERTModelEncoderF
    ( bertModel ::
        Nat ->
        Device (DeviceType Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Type
    )
    (numLayers :: Nat)
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
  BERTModelEncoderF BERTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    EncoderOnlyTransformer
      numLayers
      'BERT
      device
      BERTDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      BERTPosEncDim
      vocabDim
      typeVocabDim
      BERTDropoutP
  BERTModelEncoderF BERTModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    EncoderOnlyTransformerWithLMHead
      numLayers
      'BERT
      device
      BERTDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      BERTPosEncDim
      vocabDim
      typeVocabDim
      BERTDropoutP

instance
  ( KnownDim headDim,
    SingI headDim,
    SingI headEmbedDim,
    KnownDim embedDim,
    SingI embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    SingI inputEmbedDim,
    KnownDim vocabDim,
    KnownDim typeVocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'BERT ('Device 'CPU) BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (BERTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  where
  type
    InitializeF (BERTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim) =
      FilePath ->
      IO (BERTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  initialize filePath =
    do
      let headDim = sing @headDim
          headEmbedDim = sing @headEmbedDim
          embedDim = sing @embedDim
          inputEmbedDim = sing @inputEmbedDim
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        BERTModel <$> lookupEncoderOnlyTransformer headDim headEmbedDim embedDim inputEmbedDim bertDropoutP bertEps "bert."

instance
  ( KnownDim headDim,
    SingI headDim,
    SingI headEmbedDim,
    KnownDim embedDim,
    SingI embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    SingI inputEmbedDim,
    KnownDim vocabDim,
    KnownDim typeVocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'BERT ('Device 'CPU) BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (BERTModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  where
  type
    InitializeF (BERTModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim) =
      FilePath ->
      IO (BERTModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  initialize filePath =
    do
      let headDim = sing @headDim
          headEmbedDim = sing @headEmbedDim
          embedDim = sing @embedDim
          inputEmbedDim = sing @inputEmbedDim
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        BERTModelWithLMHead <$> lookupEncoderOnlyTransformerWithLMHead headDim headEmbedDim embedDim inputEmbedDim bertDropoutP bertEps ""

mkBERTInput ::
  forall batchDim seqDim m output.
  ( MonadFail m,
    KnownDim batchDim,
    KnownDim seqDim,
    output
      ~ Tensor
          'WithoutGradient
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
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
    (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
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