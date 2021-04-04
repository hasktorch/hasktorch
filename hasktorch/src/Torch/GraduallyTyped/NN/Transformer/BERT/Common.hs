{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.BERT.Common where

import Control.Monad.Reader (ReaderT (runReaderT))
import Data.Kind (Type)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import GHC.TypeNats (type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder, lookupEncoder)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasLookupStack)
import Torch.GraduallyTyped.NN.Transformer.Type (TensorDict, TransformerStyle (BERT), tensorDictFromPretrained)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), Size (..))

-- | BERT dType.
type BERTDType = 'Float

-- | BERT data type.
type BERTDataType = 'DataType BERTDType

-- | BERT dropout probability type.
type BERTDropoutP = Float

-- | BERT dropout rate.
-- 'dropout_rate = 0.1'
bertDropoutP :: BERTDropoutP
bertDropoutP = 0.1

-- | BERT positional encoding dimension.
type BERTPosEncDim = 'Dim ('Name "*") ('Size 32)

-- | BERT layer-norm epsilon.
-- 'layer_norm_epsilon = 1e-12'
bertEps :: Double
bertEps = 1e-12

-- | BERT maximum number of position embeddings.
-- 'max_position_embeddings = 512'
bertMaxPositionEmbeddings :: Int
bertMaxPositionEmbeddings = 512

-- | BERT Model.
newtype
  BERTModel
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
  where
  BERTModel ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim.
    BERTModelEncoderF BERTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim ->
    BERTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim
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
        Type
    )
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  BERTModelEncoderF BERTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim =
    TransformerEncoder
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
      BERTDropoutP

instance
  ( KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'BERT ('Device 'CPU) BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (BERTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim)
  where
  type
    InitializeF (BERTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim) =
      FilePath -> IO (BERTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim)
  initialize filePath =
    do
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        BERTModel <$> lookupEncoder bertDropoutP bertEps ""
