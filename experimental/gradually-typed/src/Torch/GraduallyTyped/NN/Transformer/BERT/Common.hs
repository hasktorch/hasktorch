{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.BERT.Common where

import Control.Monad.Catch (MonadThrow)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderOnly (GEncoderOnlyTransformerF, GSimplifiedEncoderOnlyTransformer (..), encoderOnlyTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkAbsPos (..), MkTransformerAttentionMask (..), MkTransformerPaddingMask (..), STransformerHead, STransformerStyle (SBERT), TransformerHead (..), TransformerStyle (BERT), mkTransformerInput)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
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

-- | Specifies the BERT model.
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
    (typeVocabDim :: Dim (Name Symbol) (Size Nat))
    (hasDropout :: HasDropout) ::
    Type
  where
  BERTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim hasDropout =
    GSimplifiedEncoderOnlyTransformer
      (GEncoderOnlyTransformerF 'BERT transformerHead numLayers gradient device BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTPosEncDim vocabDim typeVocabDim hasDropout)
      MkAbsPos
      MkTransformerPaddingMask
      (MkTransformerAttentionMask BERTDataType)

-- | Specifies the parameters of a BERT model.
--
-- - @transformerHead@: the head of the BERT model.
-- - @numLayers@: the number of layers in the BERT model.
-- - @gradient@: whether to compute the gradient of the BERT model.
-- - @device@: the computational device on which the BERT model parameters are to be allocated.
bertModelSpec ::
  forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim hasDropout.
  ( SingI headDim,
    SingI headEmbedDim,
    SingI embedDim,
    SingI inputEmbedDim,
    SingI ffnDim,
    SingI vocabDim,
    SingI typeVocabDim
  ) =>
  STransformerHead transformerHead ->
  SNat numLayers ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (BERTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim hasDropout)
bertModelSpec transformerHead numLayers gradient device hasDropout =
  GSimplifiedEncoderOnlyTransformer
    ( encoderOnlyTransformerSpec
        SBERT
        transformerHead
        numLayers
        gradient
        device
        bertDataType
        (sing @headDim)
        (sing @headEmbedDim)
        (sing @embedDim)
        (sing @inputEmbedDim)
        (sing @ffnDim)
        bertPosEncDim
        (sing @vocabDim)
        (sing @typeVocabDim)
        hasDropout
        bertDropoutP
        bertEps
    )
    MkAbsPos
    (MkTransformerPaddingMask bertPadTokenId)
    (MkTransformerAttentionMask bertDataType bertAttentionMaskBias)

mkBERTInput ::
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
mkBERTInput = mkTransformerInput bertPadTokenId
