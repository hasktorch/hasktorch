{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.BART.Common where

import Control.Monad.Catch (MonadThrow)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (GEncoderDecoderTransformerF, GSimplifiedEncoderDecoderTransformer (..), encoderDecoderTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkAbsPos (..), MkTransformerAttentionMask (..), MkTransformerCrossAttentionMask (..), MkTransformerDecoderAttentionMask (..), MkTransformerPaddingMask (..), STransformerHead, STransformerStyle (SBART), ShiftRight (..), TransformerHead (..), TransformerStyle (BART), mkTransformerInput)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | BART dType.
type BARTDType = 'Float

-- | BART dType singleton.
bartDType :: SDType BARTDType
bartDType = sing @BARTDType

-- | BART data type.
type BARTDataType = 'DataType BARTDType

-- | BART data type singleton.
bartDataType :: SDataType BARTDataType
bartDataType = sing @BARTDataType

-- | BART dropout rate.
-- 'dropout_rate = 0.1'
bartDropoutP :: Double
bartDropoutP = 0.1

-- | BART positional encoding dimension.
type BARTPosEncDim = 'Dim ('Name "*") ('Size 1026)

-- | BART positional encoding dimension singleton.
bartPosEncDim :: SDim BARTPosEncDim
bartPosEncDim = sing @BARTPosEncDim

-- | BART layer-norm epsilon.
bartEps :: Double
bartEps = 1e-5

-- | BART maximum number of position embeddings.
-- 'max_position_embeddings = 1024'
bartMaxPositionEmbeddings :: Int
bartMaxPositionEmbeddings = 1024

-- | BART padding token id.
-- 'pad_token_id = 1'
bartPadTokenId :: Int
bartPadTokenId = 1

-- | BART begin-of-sentence token id.
-- 'bos_token_id = 0'
bartBOSTokenId :: Int
bartBOSTokenId = 0

-- | BART end-of-sentence token id.
-- 'eos_token_id = 2'
bartEOSTokenId :: Int
bartEOSTokenId = 2

-- | BART attention mask bias
bartAttentionMaskBias :: Double
bartAttentionMaskBias = -10000

-- | Specifies the BART model.
type family
  BARTModelF
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
  BARTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim hasDropout =
    GSimplifiedEncoderDecoderTransformer
      (GEncoderDecoderTransformerF 'BART transformerHead numLayers numLayers gradient device BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTPosEncDim vocabDim hasDropout)
      MkAbsPos
      MkAbsPos
      MkTransformerPaddingMask
      (MkTransformerAttentionMask BARTDataType)
      (MkTransformerCrossAttentionMask BARTDataType)
      (MkTransformerDecoderAttentionMask BARTDataType)

-- | Specifies the parameters of a BART model.
--
-- - @transformerHead@: the head of the BART model.
-- - @numLayers@: the number of layers in the BART model.
-- - @gradient@: whether to compute the gradient of the BART model.
-- - @device@: the computational device on which the BART model parameters are to be allocated.
bartModelSpec ::
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
  ModelSpec (BARTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim hasDropout)
bartModelSpec transformerHead numLayers gradient device hasDropout =
  GSimplifiedEncoderDecoderTransformer
    ( encoderDecoderTransformerSpec
        SBART
        transformerHead
        numLayers
        numLayers
        gradient
        device
        bartDataType
        (sing @headDim)
        (sing @headEmbedDim)
        (sing @embedDim)
        (sing @inputEmbedDim)
        (sing @ffnDim)
        bartPosEncDim
        (sing @vocabDim)
        hasDropout
        bartDropoutP
        bartEps
    )
    (ShiftRight bartEOSTokenId)
    (ShiftRight 0)
    (MkAbsPosWithOffset 2)
    (MkAbsPosWithOffset 2)
    (MkTransformerPaddingMask bartPadTokenId)
    (MkTransformerAttentionMask bartDataType bartAttentionMaskBias)
    (MkTransformerCrossAttentionMask bartDataType bartAttentionMaskBias)
    (MkTransformerDecoderAttentionMask bartDataType bartAttentionMaskBias)

mkBARTInput ::
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
mkBARTInput = mkTransformerInput bartPadTokenId
