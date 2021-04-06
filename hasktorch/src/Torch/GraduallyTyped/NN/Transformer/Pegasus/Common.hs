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

module Torch.GraduallyTyped.NN.Transformer.Pegasus.Common where

import Control.Monad.Reader (ReaderT (runReaderT))
import Data.Kind (Type)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import GHC.TypeNats (type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasLookupDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer, SequenceToSequenceTransformerWithLMHead, lookupSequenceToSequenceTransformer, lookupSequenceToSequenceTransformerWithLMHead)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasLookupStack)
import Torch.GraduallyTyped.NN.Transformer.Type (TensorDict, TransformerStyle (Pegasus), mkTransformerInput, mkTransformerPaddingMask, tensorDictFromPretrained)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Pegasus dType.
type PegasusDType = 'Float

-- | Pegasus data type.
type PegasusDataType = 'DataType PegasusDType

-- | Pegasus dropout probability type.
type PegasusDropoutP = Float

-- | Pegasus dropout rate.
-- 'dropout_rate = 0.1'
pegasusDropoutP :: PegasusDropoutP
pegasusDropoutP = 0.1

-- | Pegasus positional encoding dimension.
type PegasusPosEncDim = 'Dim ('Name "*") ('Size 512)

-- | Pegasus layer-norm epsilon.
-- 'layer_norm_epsilon = 1e-12'
pegasusEps :: Double
pegasusEps = 1e-12

-- | Pegasus maximum number of position embeddings.
-- 'max_position_embeddings = 512'
pegasusMaxPositionEmbeddings :: Int
pegasusMaxPositionEmbeddings = 512

-- | Pegasus padding token id.
-- 'pad_token_id = 0'
pegasusPadTokenId :: Int
pegasusPadTokenId = 0

-- | Pegasus begin-of-sentence token id.
-- 'bos_token_id = 0'
pegasusBOSTokenId :: Int
pegasusBOSTokenId = 0

-- | Pegasus end-of-sentence token id.
-- 'eos_token_id = 0'
pegasusEOSTokenId :: Int
pegasusEOSTokenId = 1

-- | Pegasus model.
newtype
  PegasusModel
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  PegasusModel ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    PegasusModelSeqToSeqF PegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim ->
    PegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

-- | Pegasus model with language modelling head.
newtype
  PegasusModelWithLMHead
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  PegasusModelWithLMHead ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    PegasusModelSeqToSeqF PegasusModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim ->
    PegasusModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

type family
  PegasusModelSeqToSeqF
    ( pegasusModel ::
        Nat ->
        Device (DeviceType Nat) ->
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
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  PegasusModelSeqToSeqF PegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformer
      numLayers
      numLayers
      'Pegasus
      device
      PegasusDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      PegasusPosEncDim
      vocabDim
      PegasusDropoutP
  PegasusModelSeqToSeqF PegasusModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformerWithLMHead
      numLayers
      numLayers
      'Pegasus
      device
      PegasusDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      PegasusPosEncDim
      vocabDim
      PegasusDropoutP

instance
  ( KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'Pegasus ('Device 'CPU) PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusDropoutP (ReaderT TensorDict IO),
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers 'Pegasus ('Device 'CPU) PegasusDataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim PegasusDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (PegasusModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  where
  type
    InitializeF (PegasusModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) =
      FilePath -> IO (PegasusModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  initialize filePath =
    do
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        PegasusModel <$> lookupSequenceToSequenceTransformer pegasusDropoutP pegasusEps "model."

instance
  ( KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'Pegasus ('Device 'CPU) PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusDropoutP (ReaderT TensorDict IO),
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers 'Pegasus ('Device 'CPU) PegasusDataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim PegasusDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (PegasusModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  where
  type
    InitializeF (PegasusModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) =
      FilePath -> IO (PegasusModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  initialize filePath =
    do
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        PegasusModelWithLMHead <$> lookupSequenceToSequenceTransformerWithLMHead pegasusDropoutP pegasusEps ""

mkPegasusInput ::
  forall batchDim seqDim m output.
  ( MonadFail m,
    WithDimC batchDim (WithDimF seqDim ([[Int]] -> m output)),
    WithDimC seqDim ([[Int]] -> m output),
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
  WithDimF batchDim (WithDimF seqDim ([[Int]] -> m output))
mkPegasusInput = mkTransformerInput @batchDim @seqDim @m pegasusPadTokenId

mkPegasusPaddingMask ::
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
    (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
mkPegasusPaddingMask = mkTransformerPaddingMask pegasusPadTokenId

data PegasusInput input where
  PegasusInput ::
    forall input.
    { pegasusInput :: input
    } ->
    PegasusInput input

deriving stock instance
  ( Show input
  ) =>
  Show (PegasusInput input)