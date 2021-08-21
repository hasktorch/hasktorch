{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.NN.Transformer.Generation where

import Control.Lens (Lens)
import Control.Monad.Catch (MonadThrow)
import Control.Monad.State (MonadState (..))
import Data.Function (fix)
import Foreign.ForeignPtr (ForeignPtr)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Index.Type (Index (NegativeIndex), SIndex (..))
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (SimplifiedEncoderDecoderTransformerGenerationInput (..), SimplifiedEncoderDecoderTransformerOutput (..))
import Torch.GraduallyTyped.Prelude (Catch, pattern (:|:))
import Torch.GraduallyTyped.Prelude.List (SList (SNil))
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), SBy (..), SSelectDim (..), SelectDim (..), Shape (..))
import Torch.GraduallyTyped.Tensor.Indexing (IndexDims, IndexType (..), Indices (..), SIndexType (..), SIndices (..), (!))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (CatHListF, HasCat (..), SqueezeDimF, UnsqueezeF, sSqueezeDim, sUnsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((/=.), (==.))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mul, mulScalar, sub, subScalar)
import Torch.GraduallyTyped.Tensor.MathOperations.Reduction (ArgmaxF, MaxAllCheckF, argmax, maxAll)
import Torch.GraduallyTyped.Tensor.Type (SGetDataType (..), SGetDevice (..), SGetDim, SGetLayout (..), SGetShape (..), Tensor, TensorLike (..), sCheckedShape, sSetDataType)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.HList (HList (HNil), pattern (:.))
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Type as ATen
import Prelude hiding (all)

decode ::
  Monad m =>
  (x -> s -> m (Maybe (x, s))) ->
  x ->
  s ->
  m (x, s)
decode f x s = do
  flip fix (x, s) $ \loop (x', s') -> do
    r <- f x' s'
    case r of
      Nothing -> pure (x', s')
      Just (x'', s'') -> loop (x'', s'')

sedtOutputToInput ::
  Monad m =>
  Lens
    (SimplifiedEncoderDecoderTransformerOutput logits encoderOutput decoderInput inputPaddingMask)
    (m (SimplifiedEncoderDecoderTransformerGenerationInput decoderInput' encoderOutput inputPaddingMask))
    (logits, decoderInput)
    (m decoderInput')
sedtOutputToInput f SimplifiedEncoderDecoderTransformerOutput {..} =
  ( \decoderInput' ->
      SimplifiedEncoderDecoderTransformerGenerationInput
        <$> decoderInput' <*> pure sedtEncoderOutput <*> pure sedtInputPaddingMask
  )
    <$> f (sedtDecoderOutput, sedtOriginalDecoderInput)

prepNext ::
  ( logits ~ Tensor logitsGradient logitsLayout logitsDevice logitsDataType logitsShape,
    ntShape' ~ UnsqueezeF ('SelectDim ('ByIndex 1)) ntShape,
    Catch ntShape',
    tensors ~ '[decoderInput, Tensor ntGradient ntLayout ntDevice ntDataType ntShape'],
    decoderInput' ~ CatHListF ('SelectDim ('ByIndex 1)) tensors,
    ATen.Castable decoderInput' (ForeignPtr ATen.Tensor),
    ATen.Castable (HList tensors) (ForeignPtr ATen.TensorList),
    MonadThrow m
  ) =>
  Lens
    (logits, decoderInput)
    (m decoderInput')
    logits
    (m (Tensor ntGradient ntLayout ntDevice ntDataType ntShape))
prepNext f (logits, decoderInput) =
  ( \nextTokens -> do
      nextTokens' <- nextTokens
      nextTokens'' <- sUnsqueeze (SSelectDim $ SByIndex @1) nextTokens'
      sCat (SSelectDim $ SByIndex @1) (decoderInput :. nextTokens'' :. HNil)
  )
    <$> f logits

greedyNextTokens ::
  ( nextTokenLogitsShape ~ IndexDims ('Indices '[ 'SliceAll, 'SliceAt ('NegativeIndex 1), 'SliceAll]) logitsShape,
    nextTokensShape ~ ArgmaxF ('SelectDim ('ByIndex 1)) nextTokenLogitsShape,
    Catch nextTokensShape,
    nextTokensShape' ~ SqueezeDimF ('SelectDim ('ByIndex 1)) nextTokensShape,
    ntShape ~ 'Shape '[ntDim],
    Catch (nextTokensShape' <+> ntShape),
    SGetShape nextTokensShape',
    SGetDim ntDim,
    Catch ntDim,
    Catch nextTokensShape',
    MonadThrow m,
    MonadState (Tensor ('Gradient 'WithoutGradient) logitsLayout logitsDevice ('DataType 'Int64) ntShape) m,
    SGetDevice logitsDevice,
    SGetLayout logitsLayout
  ) =>
  Int ->
  Int ->
  Tensor logitsGradient logitsLayout logitsDevice logitsDataType logitsShape ->
  m (Tensor ('Gradient 'WithoutGradient) logitsLayout logitsDevice ('DataType 'Int64) ntShape)
greedyNextTokens padTokenId eosTokenId logits = do
  nextTokenLogits <- logits ! SIndices (SSliceAll :|: SSliceAt (SNegativeIndex @1) :|: SSliceAll :|: SNil)
  nextTokens <- argmax (SSelectDim $ SByIndex @1) nextTokenLogits
  nextTokens' <- sSqueezeDim (SSelectDim $ SByIndex @1) nextTokens
  unfinishedSequences <- get
  let usShape = sGetShape unfinishedSequences
  nextTokens'' <- sCheckedShape usShape nextTokens'
  nextTokens''' <- applyUnfinishedSequences padTokenId unfinishedSequences nextTokens''
  unfinishedSequences' <- updateUnfinishedSequences eosTokenId nextTokens''' unfinishedSequences
  put unfinishedSequences'
  pure nextTokens'''

allSequencesFinished ::
  ( SGetLayout usLayout,
    SGetDevice usDevice,
    MonadThrow m,
    Catch (usDataType <+> 'DataType 'Int64),
    Catch (BroadcastShapesF usShape ('Shape '[])),
    MaxAllCheckF usShape
  ) =>
  Tensor usGradient usLayout usDevice usDataType usShape ->
  m Bool
allSequencesFinished unfinishedSequences = do
  let gradient = SGradient SWithoutGradient
      layout = sGetLayout unfinishedSequences
      device = sGetDevice unfinishedSequences
  zero <- sToTensor gradient layout device (0 :: Int)
  isZero <- maxAll unfinishedSequences ==. zero
  pure $ fromTensor isZero

applyUnfinishedSequences ::
  ( MonadThrow m,
    kntDataType ~ (usDataType <+> ntDataType),
    kntShape ~ BroadcastShapesF usShape ntShape,
    Catch kntShape,
    ntGradient' ~ (usGradient <|> ntGradient),
    ntLayout' ~ ((usLayout <+> ntLayout) <+> usLayout),
    ntDevice' ~ ((usDevice <+> ntDevice) <+> usDevice),
    ntDataType' ~ ((usDataType <+> ntDataType) <+> usDataType),
    ntShape' ~ BroadcastShapesF kntShape usShape,
    Catch ntShape'
  ) =>
  Int ->
  Tensor usGradient usLayout usDevice usDataType usShape ->
  Tensor ntGradient ntLayout ntDevice ntDataType ntShape ->
  m (Tensor ntGradient' ntLayout' ntDevice' ntDataType' ntShape')
applyUnfinishedSequences padTokenId unfinishedSequences nextTokens = do
  keptNextTokens <- unfinishedSequences `mul` nextTokens
  replacedNextTokens <- do
    finishedSequences <- unfinishedSequences `subScalar` (1 :: Int)
    finishedSequences `mulScalar` padTokenId
  keptNextTokens `sub` replacedNextTokens

updateUnfinishedSequences ::
  ( Catch (ntDataType <+> 'DataType 'Int64),
    Catch (BroadcastShapesF ntShape ('Shape '[])),
    Catch usShape',
    SGetDataType usDataType,
    SGetDevice ntDevice,
    SGetLayout ntLayout,
    MonadThrow m,
    BroadcastShapesF usShape (BroadcastShapesF ntShape ('Shape '[])) ~ usShape',
    (usGradient <|> 'Gradient 'WithoutGradient) ~ usGradient',
    (usLayout <+> ntLayout) ~ usLayout',
    (usDevice <+> ntDevice) ~ usDevice'
  ) =>
  Int ->
  Tensor ntGradient ntLayout ntDevice ntDataType ntShape ->
  Tensor usGradient usLayout usDevice usDataType usShape ->
  m (Tensor usGradient' usLayout' usDevice' usDataType usShape')
updateUnfinishedSequences eosTokenId nextTokens unfinishedSequences = do
  let gradient = SGradient SWithoutGradient
      ntLayout = sGetLayout nextTokens
      ntDevice = sGetDevice nextTokens
      usDataType = sGetDataType unfinishedSequences
  eosTokenId' <- sToTensor gradient ntLayout ntDevice eosTokenId
  isNotEos <- nextTokens /=. eosTokenId'
  isNotEos' <- sSetDataType usDataType isNotEos
  unfinishedSequences `mul` isNotEos'
