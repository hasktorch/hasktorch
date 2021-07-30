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
import Data.Singletons.TypeLits (SNat)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderOnly (EOTEmbeddingF, EOTEncoderF, EOTHeadF, EOTTypeEmbeddingF, GEncoderOnlyTransformer, encoderOnlyTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkTransformerPaddingMaskC, STransformerHead, STransformerStyle (SBERT), TransformerHead (..), TransformerStyle (BERT), mkTransformerInput, mkTransformerPaddingMask)
import Torch.GraduallyTyped.Prelude (Seq)
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

-- | Generic BERT model data type.
data
  GBERTModel
    (bertModel :: Type)
  where
  GBERTModel ::
    forall bertModel.
    { bertModel :: bertModel
    } ->
    GBERTModel bertModel

type instance ModelSpec (GBERTModel bertModel) = GBERTModel (ModelSpec bertModel)

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
    (typeVocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  BERTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim =
    GBERTModel
      ( GEncoderOnlyTransformer
          inputEmbedDim
          (EOTEncoderF 'BERT numLayers gradient device BERTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BERTPosEncDim)
          (EOTEmbeddingF 'BERT gradient device BERTDataType inputEmbedDim vocabDim)
          (EOTTypeEmbeddingF 'BERT gradient device BERTDataType inputEmbedDim typeVocabDim)
          (EOTHeadF 'BERT transformerHead gradient device BERTDataType inputEmbedDim vocabDim)
      )

-- | Specifies the parameters of a BERT model.
--
-- - @transformerHead@: the head of the BERT model.
-- - @numLayers@: the number of layers in the BERT model.
-- - @gradient@: whether to compute the gradient of the BERT model.
-- - @device@: the computational device on which the BERT model parameters are to be allocated.
bertModelSpec ::
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
  ModelSpec (BERTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim typeVocabDim)
bertModelSpec transformerHead numLayers gradient device =
  GBERTModel $
    encoderOnlyTransformerSpec
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
      bertDropoutP
      bertEps

instance HasStateDict spec => HasStateDict (GBERTModel spec) where
  fromStateDict (GBERTModel spec) k = GBERTModel <$> fromStateDict spec k
  toStateDict k GBERTModel {..} = toStateDict k bertModel

mkBERTInput ::
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
