{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.Pegasus.Common where

import Control.Monad.Catch (MonadThrow)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (GEncoderDecoderTransformerF, GSimplifiedEncoderDecoderTransformer (..), encoderDecoderTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkAbsPos (..), MkTransformerAttentionMask (..), MkTransformerCrossAttentionMask (..), MkTransformerDecoderAttentionMask (..), MkTransformerPaddingMask (..), STransformerHead, STransformerStyle (SPegasus), ShiftRight (..), TransformerHead (..), TransformerStyle (Pegasus), mkTransformerInput)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Pegasus dType.
type PegasusDType = 'Float

-- | Pegasus dType singleton.
pegasusDType :: SDType PegasusDType
pegasusDType = sing @PegasusDType

-- | Pegasus data type.
type PegasusDataType = 'DataType PegasusDType

-- | Pegasus data type singleton.
pegasusDataType :: SDataType PegasusDataType
pegasusDataType = sing @PegasusDataType

-- | Pegasus dropout rate.
-- 'dropout_rate = 0.1'
pegasusDropoutP :: Double
pegasusDropoutP = 0.1

-- | Pegasus positional encoding dimension.
type PegasusPosEncDim = 'Dim ('Name "*") ('Size 512)

-- | Pegasus positional encoding dimension singleton.
pegasusPosEncDim :: SDim PegasusPosEncDim
pegasusPosEncDim = sing @PegasusPosEncDim

-- | Pegasus layer-norm epsilon.
pegasusEps :: Double
pegasusEps = 1e-5

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
pegasusBOSTokenId = pegasusPadTokenId

-- | Pegasus end-of-sentence token id.
-- 'eos_token_id = 0'
pegasusEOSTokenId :: Int
pegasusEOSTokenId = 1

-- | Pegasus attention mask bias
pegasusAttentionMaskBias :: Double
pegasusAttentionMaskBias = -10000

-- | Specifies the Pegasus model.
type family
  PegasusModelF
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
    (hasDropout :: HasDropout) ::
    Type
  where
  PegasusModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim hasDropout =
    GSimplifiedEncoderDecoderTransformer
      (GEncoderDecoderTransformerF 'Pegasus transformerHead numLayers numLayers gradient device PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusPosEncDim vocabDim hasDropout)
      MkAbsPos
      MkAbsPos
      MkTransformerPaddingMask
      (MkTransformerAttentionMask PegasusDataType)
      (MkTransformerCrossAttentionMask PegasusDataType)
      (MkTransformerDecoderAttentionMask PegasusDataType)

-- | Specifies the parameters of a Pegasus model.
--
-- - @transformerHead@: the head of the Pegasus model.
-- - @numLayers@: the number of layers in the Pegasus model.
-- - @gradient@: whether to compute the gradient of the Pegasus model.
-- - @device@: the computational device on which the Pegasus model parameters are to be allocated.
pegasusModelSpec ::
  forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim hasDropout.
  ( SingI headDim,
    SingI headEmbedDim,
    SingI embedDim,
    SingI inputEmbedDim,
    SingI ffnDim,
    SingI vocabDim
  ) =>
  STransformerHead transformerHead ->
  SNat numLayers ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (PegasusModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim hasDropout)
pegasusModelSpec transformerHead numLayers gradient device hasDropout =
  GSimplifiedEncoderDecoderTransformer
    ( encoderDecoderTransformerSpec
        SPegasus
        transformerHead
        numLayers
        numLayers
        gradient
        device
        pegasusDataType
        (sing @headDim)
        (sing @headEmbedDim)
        (sing @embedDim)
        (sing @inputEmbedDim)
        (sing @ffnDim)
        pegasusPosEncDim
        (sing @vocabDim)
        hasDropout
        pegasusDropoutP
        pegasusEps
    )
    (ShiftRight pegasusBOSTokenId)
    (ShiftRight 0)
    MkAbsPos
    MkAbsPos
    (MkTransformerPaddingMask pegasusPadTokenId)
    (MkTransformerAttentionMask pegasusDataType pegasusAttentionMaskBias)
    (MkTransformerCrossAttentionMask pegasusDataType pegasusAttentionMaskBias)
    (MkTransformerDecoderAttentionMask pegasusDataType pegasusAttentionMaskBias)

mkPegasusInput ::
  forall batchDim seqDim device m output.
  ( MonadThrow m,
    SGetDim batchDim,
    SGetDim seqDim,
    Catch
      ( 'Shape
          '[ 'Dim ('Name "*") 'UncheckedSize,
             'Dim ('Name "*") 'UncheckedSize
           ]
          <+> 'Shape '[batchDim, seqDim]
      ),
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          device
          ('DataType 'Int64)
          ('Shape '[batchDim, seqDim])
  ) =>
  SDim batchDim ->
  SDim seqDim ->
  SDevice device ->
  [[Int]] ->
  m output
mkPegasusInput = mkTransformerInput pegasusPadTokenId
