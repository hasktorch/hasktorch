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

import Control.Monad.State (MonadState, StateT (..), get, put, evalStateT)
import Control.Lens (Lens, Traversal, Lens', (^.), (%~))
import Control.Monad.Catch (MonadThrow)
import Data.Function (fix)
import Data.Singletons.Prelude.List (SList (SNil))
import Foreign.ForeignPtr (ForeignPtr)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Index.Type (Index (NegativeIndex), SIndex (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), stateDictFromFile, HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (SimplifiedEncoderDecoderTransformerGenerationInput (..), SimplifiedEncoderDecoderTransformerOutput (..), SimplifiedEncoderDecoderTransformerOutput' (..), SimplifiedEncoderDecoderTransformerInput' (..))
import Torch.GraduallyTyped.Prelude (Catch, pattern (:|:))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), SBy (..), SSelectDim (..), SelectDim (..), Shape (..), SShape (..), pattern SNoName, pattern (:&:), SSize (..))
import Torch.GraduallyTyped.Tensor.Indexing (IndexDims, IndexType (..), Indices (..), SIndexType (..), SIndices (..), (!))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (CatHListF, HasCat (..), SqueezeDimF, UnsqueezeF, sSqueezeDim, sUnsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((/=.), (==.))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mul, mulScalar, sub, subScalar)
import Torch.GraduallyTyped.Tensor.MathOperations.Reduction (ArgmaxF, all, argmax, sAllDim, maxAll, MaxAllCheckF)
import Torch.GraduallyTyped.Tensor.Type (TensorSpec (..), SGetDataType (..), SGetDevice (..), SGetLayout (..), Tensor, TensorLike (..), sSetDataType, SGetShape (..), sCheckedShape, SGetDim)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (..), mkTransformerInput)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.GraduallyTyped.Device (SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.NN.Transformer.BART.Common (bartPadTokenId, bartEOSTokenId)
import Torch.GraduallyTyped.NN.Transformer.BART.Base (bartBaseSpec)
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.Random (Generator, sMkGenerator)
import Torch.GraduallyTyped.NN.Type (SHasDropout (..))
import Torch.HList (HList (HNil), pattern (:.))
import qualified Torch.Internal.Class as ATen (Castable)
import qualified Torch.Internal.Type as ATen
import Prelude hiding (all)
import Control.Monad ((<=<), (>=>))
import Control.Monad.State (MonadState (..))
import Torch.GraduallyTyped.Layout (Layout(Layout), LayoutType (Dense))
import qualified Tokenizers

decode ::
  Monad m =>
  (x -> s -> m (x, s)) ->
  (x -> s -> m Bool) ->
  x ->
  s ->
  m (x, s)
decode f p x s = do
  flip fix (x, s) $ \loop (x', s') -> do
      (x'', s'') <- f x' s'
      done <- p x'' s''
      if done
        then pure (x'', s'')
        else loop (x'', s'')

greedySearch padTokenId eosTokenId model zoom =
  decode (\input g -> do
    (output, g') <- forward model input g
    input' <- (zoom . prepNext %~ greedyNextTokens padTokenId eosTokenId) output
    pure (input', g')
  ) (\_ _ -> do
    unfinishedSequences <- get
    allSequencesFinished unfinishedSequences
  )

testGreedySearch :: [String] -> IO [String]
testGreedySearch xs =
  Tokenizers.withTokenizerFromConfigFile "/tmp/bart-base-tokenizer.json" $
    \tokenizer -> do
      stateDict <- stateDictFromFile "/tmp/bart-base-state-dict.pt"

      encoderIds <- traverse (\s -> Tokenizers.encode tokenizer s >>= Tokenizers.getIDs) xs

      let device = SDevice SCPU
          padTokenId = bartPadTokenId
          eosTokenId = bartEOSTokenId
          batchDim = SNoName :&: SUncheckedSize (fromIntegral $ length encoderIds)
          seqDim = SNoName :&: SUncheckedSize (fromIntegral $ min 512 (foldr (max . length) 0 encoderIds))
      
      let spec = bartBaseSpec SWithLMHead (SGradient SWithoutGradient) device SWithoutDropout
      model <- flip evalStateT stateDict $ fromStateDict spec mempty

      g <- sMkGenerator device 0

      input <- SimplifiedEncoderDecoderTransformerInput'
                  <$> mkTransformerInput
                        padTokenId
                        batchDim
                        seqDim
                        device
                        encoderIds
      
      (SimplifiedEncoderDecoderTransformerOutput' encoderOutput paddingMask, g') <- forward model input g

      x <- SimplifiedEncoderDecoderTransformerGenerationInput 
                <$> mkTransformerInput
                      padTokenId
                      batchDim
                      (SNoName :&: SUncheckedSize 0)
                      device
                      []
                <*> pure encoderOutput
                <*> pure paddingMask

      us <- sOnes $ TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device (SDataType SInt64) (SShape $ batchDim :|: SNil)

      ((SimplifiedEncoderDecoderTransformerGenerationInput decoderInput _ _, _g), _us) <- flip runStateT us $ 
        greedySearch padTokenId eosTokenId model sedtOutputToInput x g'
      
      let decoderIds :: [[Int]] = fromTensor decoderInput

      traverse (Tokenizers.decode tokenizer) decoderIds

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
