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

module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common where

import Control.Monad.Catch (MonadThrow)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import Data.Singletons.TypeLits (SNat)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderOnly (EOTEmbeddingF, EOTEncoderF, EOTHeadF, EOTTypeEmbeddingF, GEncoderOnlyTransformer, GSimplifiedEncoderOnlyTransformer (..), encoderOnlyTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkAbsPos (..), MkTransformerAttentionMask (..), MkTransformerPaddingMask (..), STransformerHead, STransformerStyle (SRoBERTa), TransformerHead (..), TransformerStyle (RoBERTa), mkTransformerInput)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor)
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

-- | RoBERTa dropout rate.
-- 'dropout_rate = 0.1'
robertaDropoutP :: Double
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

-- | Specifies the RoBERTa model.
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
  RoBERTaModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    GSimplifiedEncoderOnlyTransformer
      ( GEncoderOnlyTransformer
          inputEmbedDim
          (EOTEncoderF 'RoBERTa numLayers gradient device RoBERTaDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim RoBERTaPosEncDim)
          (EOTEmbeddingF 'RoBERTa gradient device RoBERTaDataType inputEmbedDim vocabDim)
          (EOTTypeEmbeddingF 'RoBERTa gradient device RoBERTaDataType inputEmbedDim typeVocabDim)
          (EOTHeadF 'RoBERTa transformerHead gradient device RoBERTaDataType inputEmbedDim vocabDim)
      )
      MkAbsPos
      MkTransformerPaddingMask
      (MkTransformerAttentionMask RoBERTaDataType)

-- | Specifies the parameters of a RoBERTa model.
--
-- - @transformerHead@: the head of the RoBERTa model.
-- - @numLayers@: the number of layers in the RoBERTa model.
-- - @gradient@: whether to compute the gradient of the RoBERTa model.
-- - @device@: the computational device on which the RoBERTa model parameters are to be allocated.
robertaModelSpec ::
  forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim.
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
  ModelSpec (RoBERTaModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
robertaModelSpec transformerHead numLayers gradient device =
  GSimplifiedEncoderOnlyTransformer
    ( encoderOnlyTransformerSpec
        SRoBERTa
        transformerHead
        numLayers
        gradient
        device
        robertaDataType
        (sing @headDim)
        (sing @headEmbedDim)
        (sing @embedDim)
        (sing @inputEmbedDim)
        (sing @ffnDim)
        robertaPosEncDim
        (sing @vocabDim)
        (sing @typeVocabDim)
        robertaDropoutP
        robertaEps
    )
    (MkAbsPosWithOffset 2)
    (MkTransformerPaddingMask robertaPadTokenId)
    (MkTransformerAttentionMask robertaDataType robertaAttentionMaskBias)

mkRoBERTaInput ::
  forall batchDim seqDim device m output.
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
          device
          ('DataType 'Int64)
          ('Shape '[batchDim, seqDim])
  ) =>
  SDim batchDim ->
  SDim seqDim ->
  SDevice device ->
  [[Int]] ->
  m output
mkRoBERTaInput = mkTransformerInput robertaPadTokenId
