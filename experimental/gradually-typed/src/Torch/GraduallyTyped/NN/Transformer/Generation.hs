{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

{-# LANGUAGE RankNTypes #-}
-- {-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL #-}

module Torch.GraduallyTyped.NN.Transformer.Generation where

import Control.Lens (Lens, Traversal, Lens', (^.), (%~))
import Control.Monad.Catch (MonadThrow)
import Data.Function (fix)
import Data.Singletons.Prelude.List (SList (SNil))
import Foreign.ForeignPtr (ForeignPtr)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Index.Type (Index (NegativeIndex), SIndex (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..))
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (SimplifiedEncoderDecoderTransformerGenerationInput (..), SimplifiedEncoderDecoderTransformerOutput (..))
import Torch.GraduallyTyped.Prelude (Catch, pattern (:|:))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), SBy (..), SSelectDim (..), SelectDim (..), Shape (..))
import Torch.GraduallyTyped.Tensor.Indexing (IndexDims, IndexType (..), Indices (..), SIndexType (..), SIndices (..), (!))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (CatHListF, HasCat (..), SqueezeDimF, UnsqueezeF, sSqueezeDim, sUnsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((/=.), (==.))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mul, mulScalar, sub, subScalar)
import Torch.GraduallyTyped.Tensor.MathOperations.Reduction (ArgmaxF, all, argmax, sAllDim, maxAll, MaxAllCheckF)
import Torch.GraduallyTyped.Tensor.Type (SGetDataType (..), SGetDevice (..), SGetLayout (..), Tensor, TensorLike (..), sSetDataType, SGetShape (..), sCheckedShape, SGetDim)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.HList (HList (HNil), pattern (:.))
import qualified Torch.Internal.Class as ATen (Castable)
import qualified Torch.Internal.Type as ATen
import Prelude hiding (all)
import Control.Monad ((<=<), (>=>))
import Control.Monad.State (MonadState (..))
import Torch.GraduallyTyped.Layout (Layout(Layout), LayoutType (Dense))

decode ::
  Monad m =>
  (input -> env -> m (input', env')) ->
  (input' -> env' -> m (input', env')) ->
  (input' -> env' -> m Bool) ->
  input ->
  env ->
  m (input', env')
decode f f' p input env = do
  (input', env') <- f input env
  done <- p input' env'
  if done
    then pure (input', env')
    else flip fix (input', env') $ \loop (input'', env'') -> do
      (input''', env''') <- f' input'' env''
      done' <- p input''' env'''
      if done'
        then pure (input''', env''')
        else loop (input''', env''')

-- sedtGreedySearch padTokenId model =
--   decode (\input (unfinishedSequences, g) -> do
--     (output, g') <- forward model input g
--     input' <- bar padTokenId unfinishedSequences output
--     pure (input', (unfinishedSequences, g')))
--     (\input (unfinishedSequences, g) -> do
--     (output, g') <- forward model input g
--     input' <- bar padTokenId unfinishedSequences output
--     pure (input', (unfinishedSequences, g')))

-- Lens s (m t) a (m b)
--   forall f. Functor f => (a -> f (m b)) -> (s -> f (m t))
--
-- Lens a (m b) x (m y)
--   forall f. Functor f => (x -> f (m y)) -> (a -> f (m b))

-- foo :: Lens s (m t) a (m b) -> Lens a (m b) x (m y) -> Lens s (m t) x (m y)
-- foo f g = f . g

-- foo :: Lens' s a -> Lens' a x -> Lens' s x
-- foo f g = f . g

-- bar :: _ => _
-- bar = sedtOutputToInput . prepNext %~ greedyNextTokens

sedtOutputToInput ::
  Monad m =>
  Lens
    (SimplifiedEncoderDecoderTransformerOutput logits encoderOutput originalDecoderInput inputPaddingMask)
    (m (SimplifiedEncoderDecoderTransformerGenerationInput decoderInput' encoderOutput inputPaddingMask))
    (logits, originalDecoderInput)
    (m decoderInput')
sedtOutputToInput f SimplifiedEncoderDecoderTransformerOutput {..} =
  ( \decoderInput' ->
      SimplifiedEncoderDecoderTransformerGenerationInput
        <$> decoderInput' <*> pure sedtEncoderOutput <*> pure sedtInputPaddingMask
  )
    <$> f (sedtDecoderOutput, sedtOriginalDecoderInput)

prepNext ::
  ( logitsShape' ~ UnsqueezeF ('SelectDim ('ByIndex 1)) ntShape,
    Catch logitsShape',
    tensors ~ '[originalDecoderInput, Tensor logitsGradient logitsLayout logitsDevice logitsDataType logitsShape'],
    decoderInput' ~ CatHListF ('SelectDim ('ByIndex 1)) tensors,
    ATen.Castable decoderInput' (ForeignPtr ATen.Tensor),
    ATen.Castable (HList tensors) (ForeignPtr ATen.TensorList),
    MonadThrow m,
    logits ~ Tensor logitsGradient logitsLayout logitsDevice logitsDataType logitsShape
  ) =>
  Lens
    (logits, originalDecoderInput)
    (m decoderInput')
    logits
    (m (Tensor logitsGradient logitsLayout logitsDevice logitsDataType ntShape))
prepNext f (logits, originalDecoderInput) =
  ( \nextTokens -> do
      nextTokens' <- nextTokens
      nextTokens'' <- sUnsqueeze (SSelectDim $ SByIndex @1) nextTokens'
      sCat (SSelectDim $ SByIndex @1) (originalDecoderInput :. nextTokens'' :. HNil)
  )
    <$> f logits

greedyNextTokens ::
  ( nextTokenLogitsShape ~ IndexDims ('Indices '[ 'SliceAll, 'SliceAt ('NegativeIndex 1), 'SliceAll]) logitsShape,
    nextTokensShape ~ ArgmaxF ('SelectDim ('ByIndex 1)) nextTokenLogitsShape,
    Catch nextTokensShape,
    ntShape ~ SqueezeDimF ('SelectDim ('ByIndex 1)) nextTokensShape,
    -- ntShape ~ 'Shape '[usDim],
    usShape ~ 'Shape '[usDim],
    Catch (ntShape <+> usShape),
    SGetShape ntShape,
    SGetDim usDim,
    Catch usDim,
    Catch ntShape,
    MonadThrow m,
    MonadState (Tensor ('Gradient 'WithoutGradient) logitsLayout logitsDevice ('DataType 'Int64) usShape) m,
    ('Gradient 'WithoutGradient <|> logitsGradient) ~ logitsGradient,
    SGetDevice logitsDevice,
    SGetLayout logitsLayout
  ) =>
  Int -> 
  Int ->
  Tensor logitsGradient logitsLayout logitsDevice logitsDataType logitsShape ->
  m (Tensor logitsGradient logitsLayout logitsDevice ('DataType 'Int64) usShape)
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

-- foo padTokenId unfinishedSequences logits = do
--   nextTokens <- greedyNextTokens logits
--   applyUnfinishedSequences padTokenId unfinishedSequences nextTokens

-- bar padTokenId unfinishedSequences = 
--   let foo' = foo padTokenId unfinishedSequences
--   in sedtOutputToInput . prepNext %~ foo'

-- baz padTokenId eosTokenId unfinishedSequences logits = do

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

-- initUnfinishedSequences eosTokenId decoderInput = do
--   let gradient = SGradient SWithoutGradient
--       layout = sGetLayout decoderInput
--       device = sGetDevice decoderInput
--   eosTokenId' <- sToTensor gradient layout device eosTokenId
--   isNotEos <- decoderInput ==. eosTokenId'
--   sAllDim (SSelectDim $ SByIndex @1) isNotEos