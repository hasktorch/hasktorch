{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2 #-}

module Torch.GraduallyTyped.NN.Transformer.Type where

import Control.Monad.Catch (MonadThrow)
import Data.Singletons.TH (SingKind (fromSing), genSingletons)
import GHC.Float (double2Int)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.Prelude (Catch, forgetIsChecked, pattern (:|:))
import Torch.GraduallyTyped.Prelude.List (SList (SNil))
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (AddDimF, BroadcastShapesF, ReplaceDimF, sGetDimFromShape, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SBy (..), SDim (sDimSize), SName (..), SSelectDim (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.Creation (sArangeNaturals, sFull, sOnes, sZeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (UnsqueezeF, cat, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((==.))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (addScalar, logicalOr)
import Torch.GraduallyTyped.Tensor.Other (maskedFill, triu)
import Torch.GraduallyTyped.Tensor.Type (SGetDataType (sGetDataType), SGetDevice (..), SGetDim, SGetLayout (..), SGetShape (..), Tensor (..), TensorLike (sToTensor), TensorSpec (..), bool, sCheckedShape)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.HList

-- | A data type representing the style of a transformer.
-- Every supported transformer has a constructor of this type.
data TransformerStyle
  = -- | @T5@ transformer style, see https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
    T5
  | -- | @ByT5@ transformer style, see https://arxiv.org/abs/2105.13626
    ByT5
  | -- | @BART@ transformer style, see https://arxiv.org/abs/1910.13461
    BART
  | -- | @MBART@ transformer style, see https://arxiv.org/abs/2001.08210
    MBART
  | -- | @Pegasus@ transformer style, see https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html
    Pegasus
  | -- | @BERT@ transformer style, see https://arxiv.org/abs/1810.04805
    BERT
  | -- | @RoBERTa@ transformer style, see https://arxiv.org/abs/1907.11692
    RoBERTa
  | -- | @GPT2@ transformer style, see https://openai.com/blog/better-language-models/
    GPT2
  deriving (Show, Eq)

genSingletons [''TransformerStyle]

-- | A data type representing the type of head used in a transformer.
data TransformerHead = WithoutHead | WithLMHead

genSingletons [''TransformerHead]

padded :: Integral n => n -> a -> [a] -> [a]
padded n p xs =
  let n' = fromIntegral n
      diff = n' - length xs
   in take n' xs ++ replicate diff p

-- | Converts a doubly-nested list of input ids to a batched input tensor.
-- The outer list is over batches, the inner list over sequences.
-- The batch size is inferred from the length of the outer list.
-- The sequence length is inferred from the length of the inner list.
-- The input ids are padded to the maximum sequence length.
-- The output tensor is truncated to the maximum sequence length.
mkTransformerInput ::
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
  -- | padding token id
  Int ->
  -- | batch dimension singleton
  SDim batchDim ->
  -- | sequence dimension singleton
  SDim seqDim ->
  -- | device for the tensor
  SDevice device ->
  -- | batch of input ids
  [[Int]] ->
  -- | input tensor
  m output
mkTransformerInput padTokenId batchDim seqDim device xs =
  sToTensor gradient layout device paddedXs
    >>= sCheckedShape (SShape $ batchDim :|: seqDim :|: SNil)
  where
    gradient = SGradient SWithoutGradient
    layout = SLayout SDense
    batchSize = forgetIsChecked . dimSize $ fromSing batchDim
    seqSize = forgetIsChecked . dimSize $ fromSing seqDim
    emptySeq = replicate (fromIntegral seqSize) padTokenId
    paddedXs = padded batchSize emptySeq (padded seqSize padTokenId <$> xs)

type MkPosC device shape seqDim seqName seqSize output =
  ( SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    seqDim ~ 'Dim seqName seqSize,
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          device
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") seqSize])
  )

-- | Computes absolute positions of the input tokens.
-- Given an input tensor of shape @[batchDim, Dim seqName seqSize]@,
-- returns a tensor of shape @[Dim "*" seqSize]@.
mkPos ::
  forall m gradient layout device dataType shape seqDim seqName seqSize output.
  ( MonadThrow m,
    MkPosC device shape seqDim seqName seqSize output
  ) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | positions of the input tokens
  m output
mkPos input = do
  let device = sGetDevice input
      shape = sGetShape input
  seqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) shape
  let seqSize = sDimSize seqDim
  sArangeNaturals
    (SGradient SWithoutGradient)
    (SLayout SDense)
    device
    (SDataType SInt64)
    seqSize

data MkAbsPos = MkAbsPos | MkAbsPosWithOffset {absPosOffset :: Int}
  deriving stock (Eq, Ord, Show, Generic)

type instance ModelSpec MkAbsPos = MkAbsPos

instance HasInitialize MkAbsPos generatorDevice MkAbsPos generatorDevice where
  initialize spec g = pure (spec, g)

instance HasStateDict MkAbsPos where
  fromStateDict spec _ = pure spec
  toStateDict _ _ = pure ()

instance
  MkPosC device shape seqDim seqName seqSize output =>
  HasForward
    MkAbsPos
    (Tensor gradient layout device dataType shape)
    generatorDevice
    (Tensor ('Gradient 'WithoutGradient) ('Layout 'Dense) device ('DataType 'Int64) ('Shape '[ 'Dim ('Name "*") seqSize]))
    generatorDevice
  where
  forward MkAbsPos input g = do
    pos <- mkPos input
    pure (pos, g)
  forward MkAbsPosWithOffset {..} input g = do
    pos <- mkPos input
    pos' <- pos `addScalar` absPosOffset
    pure (pos', g)

-- | Computes relative positions of the input tokens to the encoder.
--
-- >>> mkRelPos' 32 128 21 17
-- [[0,17,18,19,20,21,22,23,24,24,24,24,25,25,25,25,26],[1,0,17,18,19,20,21,22,23,24,24,24,24,25,25,25,25],[2,1,0,17,18,19,20,21,22,23,24,24,24,24,25,25,25],[3,2,1,0,17,18,19,20,21,22,23,24,24,24,24,25,25],[4,3,2,1,0,17,18,19,20,21,22,23,24,24,24,24,25],[5,4,3,2,1,0,17,18,19,20,21,22,23,24,24,24,24],[6,5,4,3,2,1,0,17,18,19,20,21,22,23,24,24,24],[7,6,5,4,3,2,1,0,17,18,19,20,21,22,23,24,24],[8,7,6,5,4,3,2,1,0,17,18,19,20,21,22,23,24],[8,8,7,6,5,4,3,2,1,0,17,18,19,20,21,22,23],[8,8,8,7,6,5,4,3,2,1,0,17,18,19,20,21,22],[8,8,8,8,7,6,5,4,3,2,1,0,17,18,19,20,21],[9,8,8,8,8,7,6,5,4,3,2,1,0,17,18,19,20],[9,9,8,8,8,8,7,6,5,4,3,2,1,0,17,18,19],[9,9,9,8,8,8,8,7,6,5,4,3,2,1,0,17,18],[9,9,9,9,8,8,8,8,7,6,5,4,3,2,1,0,17],[10,9,9,9,9,8,8,8,8,7,6,5,4,3,2,1,0],[10,10,9,9,9,9,8,8,8,8,7,6,5,4,3,2,1],[10,10,10,9,9,9,9,8,8,8,8,7,6,5,4,3,2],[10,10,10,10,9,9,9,9,8,8,8,8,7,6,5,4,3],[10,10,10,10,10,9,9,9,9,8,8,8,8,7,6,5,4]]
mkRelPos' :: Int -> Int -> Int -> Int -> [[Int]]
mkRelPos' numBuckets maxDistance querySize keySize =
  let queryPos = [0, 1 .. querySize - 1]
      keyPos = [0, 1 .. keySize - 1]
      numBuckets' = numBuckets `div` 2
      maxExact = numBuckets' `div` 2
   in fmap
        ( \qp ->
            fmap
              ( \kp ->
                  let rawRelPos = kp - qp
                      absRelPos = abs rawRelPos
                      relBucket = if rawRelPos > 0 then numBuckets' else 0
                      relBucket' =
                        let isSmall = absRelPos < maxExact
                            relPosIfLarge =
                              maxExact
                                + double2Int
                                  ( logBase
                                      (fromIntegral maxDistance / fromIntegral maxExact)
                                      (fromIntegral absRelPos / fromIntegral maxExact)
                                      * fromIntegral (numBuckets' - maxExact)
                                  )
                            relPosIfLarge' = min relPosIfLarge (numBuckets' - 1)
                         in if isSmall then absRelPos else relPosIfLarge'
                   in relBucket + relBucket'
              )
              keyPos
        )
        queryPos

type MkRelPosC device shape seqDim seqName seqSize output =
  ( SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    seqDim ~ 'Dim seqName seqSize,
    Catch
      ( '[ 'Dim ('Name "*") 'UncheckedSize,
           'Dim ('Name "*") 'UncheckedSize
         ]
          <+> '[ 'Dim ('Name "*") seqSize, 'Dim ('Name "*") seqSize]
      ),
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          device
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") seqSize, 'Dim ('Name "*") seqSize])
  )

-- | Computes relative positions of the input tokens to the encoder.
-- Given an input tensor of shape @[batchDim, Dim seqName seqSize]@,
-- returns a tensor of shape @[1, Dim "*" seqSize, Dim "*" seqSize]@.
mkRelPos ::
  forall m gradient layout device dataType shape relPosEncBucketDim seqDim seqName seqSize output.
  ( MonadThrow m,
    MkRelPosC device shape seqDim seqName seqSize output
  ) =>
  -- | bucket dimension
  SDim relPosEncBucketDim ->
  -- | maximum distance
  Int ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | relative positions of the input tokens
  m output
mkRelPos relPosEncBucketDim maxDistance input = do
  seqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) $ sGetShape input
  let seqSize = fromInteger . forgetIsChecked . fromSing $ sDimSize seqDim
  sToTensor gradient' layout' device [mkRelPos' relPosEncBucketSize maxDistance seqSize seqSize]
    >>= sCheckedShape (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: sDimSize seqDim :|: SName @"*" :&: sDimSize seqDim :|: SNil)
  where
    gradient' = SGradient SWithoutGradient
    layout' = SLayout SDense
    device = sGetDevice input
    relPosEncBucketSize = fromInteger . forgetIsChecked . dimSize . fromSing $ relPosEncBucketDim

-- | Computes relative positions of the input tokens to the decoder.
--
-- >>> mkDecoderRelPos' 32 128 21 17
-- [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0],[6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0],[7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0],[8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0],[9,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0],[10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0],[11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0],[12,11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0],[13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,0,0],[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,0],[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0],[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],[16,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1],[16,16,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2],[17,16,16,16,15,14,13,12,11,10,9,8,7,6,5,4,3],[17,17,16,16,16,15,14,13,12,11,10,9,8,7,6,5,4]]
mkDecoderRelPos' :: Int -> Int -> Int -> Int -> [[Int]]
mkDecoderRelPos' numBuckets maxDistance querySize keySize =
  let queryPos = [0, 1 .. querySize - 1]
      keyPos = [0, 1 .. keySize - 1]
      maxExact = numBuckets `div` 2
   in fmap
        ( \qp ->
            fmap
              ( \kp ->
                  let rawRelPos = kp - qp
                      absRelPos = negate . min 0 $ rawRelPos
                      relBucket' =
                        let isSmall = absRelPos < maxExact
                            relPosIfLarge =
                              maxExact
                                + double2Int
                                  ( logBase
                                      (fromIntegral maxDistance / fromIntegral maxExact)
                                      (fromIntegral absRelPos / fromIntegral maxExact)
                                      * fromIntegral (numBuckets - maxExact)
                                  )
                            relPosIfLarge' = min relPosIfLarge (numBuckets - 1)
                         in if isSmall then absRelPos else relPosIfLarge'
                   in relBucket'
              )
              keyPos
        )
        queryPos

-- | Computes relative positions of the input tokens to the decoder.
-- Given an input tensor of shape @[batchDim, Dim seqName seqSize]@,
-- returns a tensor of shape @[1, Dim "*" seqSize, Dim "*" seqSize]@.
mkDecoderRelPos ::
  forall m gradient layout device dataType shape relPosEncBucketDim seqDim seqName seqSize output.
  ( MonadThrow m,
    MkRelPosC device shape seqDim seqName seqSize output
  ) =>
  -- | bucket dimension
  SDim relPosEncBucketDim ->
  -- | maximum distance
  Int ->
  -- | decoder input tensor
  Tensor gradient layout device dataType shape ->
  -- | relative positions of the input tokens
  m output
mkDecoderRelPos relPosEncBucketDim maxDistance input = do
  seqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) $ sGetShape input
  let seqSize = fromInteger . forgetIsChecked . fromSing $ sDimSize seqDim
  sToTensor gradient' layout' device [mkDecoderRelPos' relPosEncBucketSize maxDistance seqSize seqSize]
    >>= sCheckedShape (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: sDimSize seqDim :|: SName @"*" :&: sDimSize seqDim :|: SNil)
  where
    gradient' = SGradient SWithoutGradient
    layout' = SLayout SDense
    device = sGetDevice input
    relPosEncBucketSize = fromInteger . forgetIsChecked . dimSize . fromSing $ relPosEncBucketDim

data MkRelPos (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat)) where
  MkRelPos ::
    forall relPosEncBucketDim.
    { relPosEncBucketDim :: SDim relPosEncBucketDim,
      relPosMaxDistance :: Int
    } ->
    MkRelPos relPosEncBucketDim
  MkDecoderRelPos ::
    forall relPosEncBucketDim.
    { decoderRelPosEncBucketDim :: SDim relPosEncBucketDim,
      decoderRelPosMaxDistance :: Int
    } ->
    MkRelPos relPosEncBucketDim
  deriving stock (Show, Generic)

type instance ModelSpec (MkRelPos relPosEncBucketDim) = MkRelPos relPosEncBucketDim

instance HasInitialize (MkRelPos relPosEncBucketDim) generatorDevice (MkRelPos relPosEncBucketDim) generatorDevice where
  initialize spec g = pure (spec, g)

instance HasStateDict (MkRelPos relPosEncBucketDim) where
  fromStateDict spec _ = pure spec
  toStateDict _ _ = pure ()

instance
  MkRelPosC device shape seqDim seqName seqSize output =>
  HasForward
    (MkRelPos relPosEncBucketDim)
    (Tensor gradient layout device dataType shape)
    generatorDevice
    ( Tensor
        ('Gradient 'WithoutGradient)
        ('Layout 'Dense)
        device
        ('DataType 'Int64)
        ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") seqSize, 'Dim ('Name "*") seqSize])
    )
    generatorDevice
  where
  forward MkRelPos {..} input g = do
    relPos <- mkRelPos relPosEncBucketDim relPosMaxDistance input
    pure (relPos, g)
  forward MkDecoderRelPos {..} input g = do
    decoderRelPos <- mkDecoderRelPos decoderRelPosEncBucketDim decoderRelPosMaxDistance input
    pure (decoderRelPos, g)

type MkTransformerPaddingMaskC layout device dataType shape output =
  ( SGetDevice device,
    Catch (dataType <+> 'DataType 'Int64),
    Catch (BroadcastShapesF shape ('Shape '[])),
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          ('DataType 'Bool)
          (BroadcastShapesF shape ('Shape '[]))
  )

-- | Computes the padding mask for a transformer.
-- Given an input tensor of shape @[batchDim, Dim seqName seqSize]@,
-- returns a tensor of shape @[batchDim, Dim "*" seqSize]@.
mkTransformerPaddingMask ::
  forall m gradient layout device dataType shape output.
  ( MonadThrow m,
    MkTransformerPaddingMaskC layout device dataType shape output
  ) =>
  -- | padding token id
  Int ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | padding mask
  m output
mkTransformerPaddingMask padTokenId input = do
  let device = sGetDevice input
  padToken <- sToTensor (SGradient SWithoutGradient) (SLayout SDense) device padTokenId
  input ==. padToken

newtype MkTransformerPaddingMask = MkTransformerPaddingMask {padTokenId :: Int}
  deriving stock (Show, Generic)

type instance
  ModelSpec MkTransformerPaddingMask =
    MkTransformerPaddingMask

instance
  HasInitialize
    MkTransformerPaddingMask
    generatorDevice
    MkTransformerPaddingMask
    generatorDevice
  where
  initialize spec g = pure (spec, g)

instance HasStateDict MkTransformerPaddingMask where
  fromStateDict spec _ = pure spec
  toStateDict _ _ = pure ()

instance
  MkTransformerPaddingMaskC layout device dataType shape output =>
  HasForward
    MkTransformerPaddingMask
    (Tensor gradient layout device dataType shape)
    generatorDevice
    output
    generatorDevice
  where
  forward MkTransformerPaddingMask {..} input g = do
    paddingMask <- mkTransformerPaddingMask padTokenId input
    pure (paddingMask, g)

type MkTransformerAttentionMaskC transformerDataType gradient layout device dataType shape seqDim output =
  ( SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    Catch (gradient <+> 'Gradient 'WithoutGradient),
    Catch (dataType <+> 'DataType 'Bool),
    Catch (UnsqueezeF ('SelectDim ('ByIndex 1)) shape),
    Catch
      ( BroadcastShapesF
          (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
          ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
      ),
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          transformerDataType
          ( BroadcastShapesF
              (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
              ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
          )
  )

-- | Creates a bidirectional attention mask for a transformer.
-- Given a padding mask of shape @[batchDim, seqDim]@,
-- returns a tensor of shape @[batchDim, seqDim, seqDim]@.
mkTransformerAttentionMask ::
  forall m transformerDataType gradient layout device dataType shape seqDim output.
  ( MonadThrow m,
    MkTransformerAttentionMaskC transformerDataType gradient layout device dataType shape seqDim output
  ) =>
  -- | data type singleton of the transformer
  SDataType transformerDataType ->
  -- | attention mask bias (typically a large negative number)
  Double ->
  -- | encoder padding mask
  Tensor gradient layout device dataType shape ->
  m output
mkTransformerAttentionMask transformerDataType attentionMaskBias paddingMask = do
  let pmLayout = sGetLayout paddingMask
      pmDevice = sGetDevice paddingMask
      pmShape = sGetShape paddingMask
  pmSeqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) pmShape
  emptyMask <-
    sZeros $
      TensorSpec
        (SGradient SWithoutGradient)
        pmLayout
        pmDevice
        transformerDataType
        (SShape $ SName @"*" :&: SSize @1 :|: pmSeqDim :|: pmSeqDim :|: SNil)
  paddingMask' <- unsqueeze @('SelectDim ('ByIndex 1)) paddingMask
  maskedFill paddingMask' attentionMaskBias emptyMask

data MkTransformerAttentionMask (dataType :: DataType DType) where
  MkTransformerAttentionMask ::
    forall dataType.
    { attentionMaskDataType :: SDataType dataType,
      attentionMaskBias :: Double
    } ->
    MkTransformerAttentionMask dataType
  deriving stock (Show, Generic)

type instance
  ModelSpec (MkTransformerAttentionMask dataType) =
    MkTransformerAttentionMask dataType

instance
  HasInitialize
    (MkTransformerAttentionMask dataType)
    generatorDevice
    (MkTransformerAttentionMask dataType)
    generatorDevice
  where
  initialize spec g = pure (spec, g)

instance HasStateDict (MkTransformerAttentionMask dataType) where
  fromStateDict spec _ = pure spec
  toStateDict _ _ = pure ()

instance
  MkTransformerAttentionMaskC dataType inputGradient inputLayout inputDevice inputDataType inputShape seqDim output =>
  HasForward
    (MkTransformerAttentionMask dataType)
    (Tensor inputGradient inputLayout inputDevice inputDataType inputShape)
    generatorDevice
    output
    generatorDevice
  where
  forward MkTransformerAttentionMask {..} input g = do
    attentionMask <- mkTransformerAttentionMask attentionMaskDataType attentionMaskBias input
    pure (attentionMask, g)

type MkTransformerDecoderAttentionMaskC transformerDataType layout device shape seqDim output =
  ( SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    Catch seqDim,
    Catch (UnsqueezeF ('SelectDim ('ByIndex 1)) shape),
    Catch
      ( BroadcastShapesF
          ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
          (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
      ),
    Catch
      ( BroadcastShapesF
          ( BroadcastShapesF
              ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
              (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
          )
          ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
      ),
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          transformerDataType
          ( BroadcastShapesF
              ( BroadcastShapesF
                  ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
                  (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
              )
              ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
          )
  )

-- | Creates a causal attention mask for a transformer decoder.
-- Given a padding mask of shape @[batchDim, seqDim]@,
-- returns a tensor of shape @[batchDim, seqDim, seqDim]@.
mkTransformerDecoderAttentionMask ::
  forall m transformerDataType gradient layout device dataType shape seqDim output.
  ( MonadThrow m,
    MkTransformerDecoderAttentionMaskC transformerDataType layout device shape seqDim output
  ) =>
  -- | data type singleton of the transformer
  SDataType transformerDataType ->
  -- | attention mask bias (typically a large negative number)
  Double ->
  -- | decoder padding mask
  Tensor gradient layout device dataType shape ->
  m output
mkTransformerDecoderAttentionMask transformerDataType attentionMaskBias paddingMask = do
  let pmLayout = sGetLayout paddingMask
      pmDevice = sGetDevice paddingMask
      pmShape = sGetShape paddingMask
  pmSeqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) pmShape
  causalMask <-
    bool . triu 1
      =<< sOnes
        ( TensorSpec
            (SGradient SWithoutGradient)
            pmLayout
            pmDevice
            transformerDataType
            (SShape $ pmSeqDim :|: pmSeqDim :|: SNil)
        )
  causalMask' <- unsqueeze @('SelectDim ('ByIndex 0)) causalMask
  emptyMask <-
    sZeros $
      TensorSpec
        (SGradient SWithoutGradient)
        pmLayout
        pmDevice
        transformerDataType
        (SShape $ SName @"*" :&: SSize @1 :|: pmSeqDim :|: pmSeqDim :|: SNil)
  paddingMask' <- unsqueeze @('SelectDim ('ByIndex 1)) paddingMask
  booleanMask <- causalMask' `logicalOr` paddingMask'
  maskedFill
    booleanMask
    attentionMaskBias
    emptyMask

data MkTransformerDecoderAttentionMask (dataType :: DataType DType) where
  MkTransformerDecoderAttentionMask ::
    forall dataType.
    { decoderAttentionMaskDataType :: SDataType dataType,
      decoderAttentionMaskBias :: Double
    } ->
    MkTransformerDecoderAttentionMask dataType
  deriving stock (Show, Generic)

type instance
  ModelSpec (MkTransformerDecoderAttentionMask dataType) =
    MkTransformerDecoderAttentionMask dataType

instance
  HasInitialize
    (MkTransformerDecoderAttentionMask dataType)
    generatorDevice
    (MkTransformerDecoderAttentionMask dataType)
    generatorDevice
  where
  initialize spec g = pure (spec, g)

instance HasStateDict (MkTransformerDecoderAttentionMask dataType) where
  fromStateDict spec _ = pure spec
  toStateDict _ _ = pure ()

instance
  MkTransformerDecoderAttentionMaskC dataType decoderInputLayout decoderInputDevice decoderInputShape seqDim output =>
  HasForward
    (MkTransformerDecoderAttentionMask dataType)
    (Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
    generatorDevice
    output
    generatorDevice
  where
  forward MkTransformerDecoderAttentionMask {..} input g = do
    decoderAttentionMask <- mkTransformerDecoderAttentionMask decoderAttentionMaskDataType decoderAttentionMaskBias input
    pure (decoderAttentionMask, g)

type MkTransformerCrossAttentionMaskC transformerDataType decoderInputShape decoderInputSeqDim gradient layout device dataType shape seqDim output =
  ( SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    SGetShape decoderInputShape,
    decoderInputSeqDim ~ (decoderInputShape ! 1),
    Catch (gradient <+> 'Gradient 'WithoutGradient),
    Catch (dataType <+> 'DataType 'Bool),
    Catch (UnsqueezeF ('SelectDim ('ByIndex 1)) shape),
    Catch
      ( BroadcastShapesF
          (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
          ( 'Shape
              '[ 'Dim ('Name "*") ('Size 1), decoderInputSeqDim, seqDim]
          )
      ),
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          transformerDataType
          ( BroadcastShapesF
              (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
              ('Shape '[ 'Dim ('Name "*") ('Size 1), decoderInputSeqDim, seqDim])
          )
  )

-- | Creates a cross-attention mask for an encoder-decoder transformer.
-- Given an encoder padding mask of shape @[batchDim, seqDim]@,
-- and the shape @[batchDim, decoderSeqDim]@ of the decoder's input,
-- returns a tensor of shape @[batchDim, decoderSeqDim, seqDim]@.
mkTransformerCrossAttentionMask ::
  forall m transformerDataType decoderInputShape decoderInputSeqDim gradient layout device dataType shape seqDim output.
  ( MonadThrow m,
    MkTransformerCrossAttentionMaskC transformerDataType decoderInputShape decoderInputSeqDim gradient layout device dataType shape seqDim output
  ) =>
  -- | data type singleton of the transformer
  SDataType transformerDataType ->
  -- | decoder input shape
  SShape decoderInputShape ->
  -- | attention mask bias (typically a large negative number)
  Double ->
  -- | encoder padding mask
  Tensor gradient layout device dataType shape ->
  m output
mkTransformerCrossAttentionMask transformerDataType decoderInputShape attentionMaskBias paddingMask = do
  decoderInputSeqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) decoderInputShape
  let pmLayout = sGetLayout paddingMask
      pmDevice = sGetDevice paddingMask
      pmShape = sGetShape paddingMask
  pmSeqDim <- sGetDimFromShape (SSelectDim $ SByIndex @1) pmShape
  emptyMask <-
    sZeros $
      TensorSpec
        (SGradient SWithoutGradient)
        pmLayout
        pmDevice
        transformerDataType
        (SShape $ SName @"*" :&: SSize @1 :|: decoderInputSeqDim :|: pmSeqDim :|: SNil)
  paddingMask' <- unsqueeze @('SelectDim ('ByIndex 1)) paddingMask
  maskedFill paddingMask' attentionMaskBias emptyMask

data MkTransformerCrossAttentionMask (dataType :: DataType DType) where
  MkTransformerCrossAttentionMask ::
    forall dataType.
    { crossAttentionMaskDataType :: SDataType dataType,
      crossAttentionMaskBias :: Double
    } ->
    MkTransformerCrossAttentionMask dataType
  deriving stock (Show, Generic)

type instance
  ModelSpec (MkTransformerCrossAttentionMask dataType) =
    MkTransformerCrossAttentionMask dataType

instance
  HasInitialize
    (MkTransformerCrossAttentionMask dataType)
    generatorDevice
    (MkTransformerCrossAttentionMask dataType)
    generatorDevice
  where
  initialize spec g = pure (spec, g)

instance HasStateDict (MkTransformerCrossAttentionMask dataType) where
  fromStateDict spec _ = pure spec
  toStateDict _ _ = pure ()

instance
  MkTransformerCrossAttentionMaskC dataType decoderInputShape decoderInputSeqDim inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaksDataType inputPaddingMaskShape seqDim output =>
  HasForward
    (MkTransformerCrossAttentionMask dataType)
    ( Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaksDataType inputPaddingMaskShape
    )
    generatorDevice
    output
    generatorDevice
  where
  forward MkTransformerCrossAttentionMask {..} (decoderInput, inputPaddingMask) g = do
    let decoderInputShape = sGetShape decoderInput
    crossAttentionMask <- mkTransformerCrossAttentionMask crossAttentionMaskDataType decoderInputShape crossAttentionMaskBias inputPaddingMask
    pure (crossAttentionMask, g)

data ShiftRight fillValue where
  ShiftRight ::
    forall fillValue.
    -- | fill value for shift right
    fillValue ->
    ShiftRight fillValue
  deriving stock (Eq, Ord, Show, Generic)

type instance ModelSpec (ShiftRight fillValue) = ShiftRight fillValue

instance
  HasInitialize
    (ShiftRight fillValue)
    generatorDevice
    (ShiftRight fillValue)
    generatorDevice
  where
  initialize spec = pure . (spec,)

instance HasStateDict (ShiftRight fillValue) where
  fromStateDict spec _ = pure spec
  toStateDict _ _ = pure ()

instance
  ( input
      ~ Tensor
          inputGradient
          inputLayout
          inputDevice
          inputDataType
          inputShape,
    SGetLayout inputLayout,
    SGetDevice inputDevice,
    SGetDataType inputDataType,
    SGetShape inputShape,
    inputBatchDim ~ (inputShape ! 0),
    inputSeqDim ~ (inputShape ! 1),
    Scalar fillValue,
    rightShiftedInput
      ~ Tensor
          (inputGradient <|> 'Gradient 'WithoutGradient)
          inputLayout
          inputDevice
          inputDataType
          ( ReplaceDimF
              ('SelectDim ('ByIndex 1))
              (inputShape <+> 'Shape '[inputBatchDim, inputSeqDim])
              (AddDimF inputSeqDim ('Dim ('Name "*") ('Size 1)))
          )
  ) =>
  HasForward (ShiftRight fillValue) input generator rightShiftedInput generator
  where
  forward (ShiftRight fillValue) input g = do
    let inputLayout = sGetLayout input
        inputDevice = sGetDevice input
        inputDataType = sGetDataType input
        inputShape = sGetShape input
    inputBatchDim <- sGetDimFromShape (SSelectDim $ SByIndex @0) inputShape
    filler <-
      sFull
        ( TensorSpec
            (SGradient SWithoutGradient)
            inputLayout
            inputDevice
            inputDataType
            (SShape $ inputBatchDim :|: SName @"*" :&: SSize @1 :|: SNil)
        )
        fillValue
    shifted <- cat @('SelectDim ('ByIndex 1)) (filler :. input :. HNil)
    pure (shifted, g)
