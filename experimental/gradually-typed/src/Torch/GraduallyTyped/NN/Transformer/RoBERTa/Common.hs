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
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common where

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
import Torch.GraduallyTyped.NN.Transformer.EncoderOnly (EncoderOnlyTransformer (..), EncoderOnlyTransformerWithLMHead (..), lookupEncoderOnlyTransformer, lookupEncoderOnlyTransformerWithLMHead)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasLookupStack)
import Torch.GraduallyTyped.NN.Transformer.Type (TensorDict, TransformerStyle (RoBERTa), mkTransformerInput, mkTransformerPaddingMask, tensorDictFromPretrained)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | RoBERTa dType.
type RoBERTaDType = 'Float

-- | RoBERTa dType.
robertaDType :: SDType RoBERTaDType
robertaDType = SFloat

-- | RoBERTa data type.
type RoBERTaDataType = 'DataType RoBERTaDType

-- | RoBERTa data type.
robertaDataType :: SDataType RoBERTaDataType
robertaDataType = SDataType robertaDType

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

-- | RoBERTa model.
newtype
  RoBERTaModel
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
  RoBERTaModel ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim.
    RoBERTaModelEncoderF RoBERTaModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim ->
    RoBERTaModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim
  deriving stock (Generic)

-- | RoBERTa model with language modelling head.
newtype
  RoBERTaModelWithLMHead
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
  RoBERTaModelWithLMHead ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim.
    RoBERTaModelEncoderF RoBERTaModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim ->
    RoBERTaModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim
  deriving stock (Generic)

type family
  RoBERTaModelEncoderF
    ( robertaModel ::
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
  RoBERTaModelEncoderF RoBERTaModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    EncoderOnlyTransformer
      numLayers
      'RoBERTa
      device
      RoBERTaDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      RoBERTaPosEncDim
      vocabDim
      typeVocabDim
      RoBERTaDropoutP
  RoBERTaModelEncoderF RoBERTaModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    EncoderOnlyTransformerWithLMHead
      numLayers
      'RoBERTa
      device
      RoBERTaDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      RoBERTaPosEncDim
      vocabDim
      typeVocabDim
      RoBERTaDropoutP

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
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'RoBERTa ('Device 'CPU) RoBERTaDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim RoBERTaDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (RoBERTaModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  where
  type
    InitializeF (RoBERTaModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim) =
      FilePath -> IO (RoBERTaModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  initialize filePath =
    do
      let headDim = sing @headDim
          headEmbedDim = sing @headEmbedDim
          embedDim = sing @embedDim
          inputEmbedDim = sing @inputEmbedDim
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        RoBERTaModel <$> lookupEncoderOnlyTransformer headDim headEmbedDim embedDim inputEmbedDim robertaDropoutP robertaEps "roberta."

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
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'RoBERTa ('Device 'CPU) RoBERTaDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim RoBERTaDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (RoBERTaModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  where
  type
    InitializeF (RoBERTaModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim) =
      FilePath -> IO (RoBERTaModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
  initialize filePath =
    do
      let headDim = sing @headDim
          headEmbedDim = sing @headEmbedDim
          embedDim = sing @embedDim
          inputEmbedDim = sing @inputEmbedDim
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        RoBERTaModelWithLMHead <$> lookupEncoderOnlyTransformerWithLMHead headDim headEmbedDim embedDim inputEmbedDim robertaDropoutP robertaEps ""

mkRoBERTaInput ::
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
mkRoBERTaInput = mkTransformerInput robertaPadTokenId

mkRoBERTaPaddingMask ::
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
    (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
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