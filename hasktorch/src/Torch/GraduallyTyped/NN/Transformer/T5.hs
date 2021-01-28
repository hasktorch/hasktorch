{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2
                -fomit-interface-pragmas
                -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}

module Torch.GraduallyTyped.NN.Transformer.T5 where

import Control.Monad (guard)
import Control.Monad.Reader (ReaderT (runReaderT), ask, lift)
import qualified Data.Map as Map
import GHC.Float (double2Int)
import GHC.TypeLits (KnownNat, Nat)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Block (TransformerBlock (TransformerBlock))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention (CrossAttention))
import Torch.GraduallyTyped.NN.Transformer.Decoder (TransformerDecoder (TransformerDecoder))
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (TransformerDecoderBlock (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (TransformerDecoderStack (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder (..))
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (MultiHeadAttention (..))
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..), SequenceToSequenceTransformer (..))
import Torch.GraduallyTyped.NN.Transformer.Stack (TransformerStack (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator, mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (NumelDimF, NumelF, Shape (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), Name (..), SelectDim (..), Size (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor.Creation (ones, zeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (UnsqueezeF, unsqueeze)
import Torch.GraduallyTyped.Tensor.Other (maskedFill, triu)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), bool, checkedDataType, checkedDevice, checkedLayout, checkedShape, shape)
import Torch.Script (IValue (..))
import Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe), asTensor)

-- | num_layers = 6
type T5SmallNumLayers = 6

-- | n_heads = 8
type T5SmallHeadDim = 'Dim ( 'Name "*") ( 'Size 8)

-- | d_kv = 64
type T5SmallHeadEmbedDim = 'Dim ( 'Name "*") ( 'Size 64)

-- | inner_dim =  = n_heads * d_kv = 512
type T5SmallEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | d_model = 512
type T5SmallInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | d_ff = 2048
type T5SmallFFNDim = 'Dim ( 'Name "*") ( 'Size 2048)

-- | relative_attention_num_buckets = 32
type T5SmallRelPosEncBucketDim = 'Dim ( 'Name "*") ( 'Size 32)

-- | vocab_size = 32128
type T5SmallVocabDim = 'Dim ( 'Name "*") ( 'Size 32128)

-- | https://huggingface.co/t5-small/blob/main/config.json
data T5Small device dataType where
  T5Small ::
    forall device dataType.
    SequenceToSequenceTransformer
      'WithLMHead
      T5SmallNumLayers
      T5SmallNumLayers
      device
      dataType
      T5SmallHeadDim
      T5SmallHeadEmbedDim
      T5SmallEmbedDim
      T5SmallInputEmbedDim
      T5SmallFFNDim
      T5SmallRelPosEncBucketDim
      T5SmallVocabDim
      Float ->
    T5Small device dataType

instance
  HasForward
    (SequenceToSequenceTransformer 'WithLMHead T5SmallNumLayers T5SmallNumLayers device dataType T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallRelPosEncBucketDim T5SmallVocabDim Float)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    (Generator generatorDevice) =>
  HasForward
    (T5Small device dataType)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    (Generator generatorDevice)
  where
  type
    ForwardOutput
      (T5Small device dataType)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardOutput
        (SequenceToSequenceTransformer 'WithLMHead T5SmallNumLayers T5SmallNumLayers device dataType T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallRelPosEncBucketDim T5SmallVocabDim Float)
        ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
          Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
          Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
          Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
          Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
          Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        (Generator generatorDevice)
  type
    ForwardGeneratorOutput
      (T5Small device dataType)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardGeneratorOutput
        (SequenceToSequenceTransformer 'WithLMHead T5SmallNumLayers T5SmallNumLayers device dataType T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallRelPosEncBucketDim T5SmallVocabDim Float)
        ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
          Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
          Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
          Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
          Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
          Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        (Generator generatorDevice)
  forward (T5Small seqToSeq) inputs = forward seqToSeq inputs

-- | dropout_rate = 0.1
t5SmallDropoutP :: Float
t5SmallDropoutP = 0.1

-- | layer_norm_epsilon = 1e-06
t5SmallEps :: Double
t5SmallEps = 1e-6

type T5SmallAttentionMask device dataType batchSize inputSeqSize =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

t5SmallAttentionMask ::
  forall device dataType inputSeqSize.
  ( WithDeviceC device (WithDataTypeF dataType (T5SmallAttentionMask device dataType 1 inputSeqSize)),
    WithDataTypeC dataType (T5SmallAttentionMask device dataType 1 inputSeqSize),
    KnownNat inputSeqSize
  ) =>
  WithDeviceF device (WithDataTypeF dataType (T5SmallAttentionMask device dataType 1 inputSeqSize))
t5SmallAttentionMask = zeros @ 'WithoutGradient @( 'Layout 'Dense) @device @dataType @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size inputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

type T5SmallDecoderAttentionMask device dataType batchSize decoderInputSeqSize =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])

t5SmallDecoderAttentionMask ::
  forall device dataType decoderInputSeqSize.
  ( WithDeviceC device (WithDataTypeF dataType (T5SmallDecoderAttentionMask device dataType 1 decoderInputSeqSize)),
    WithDataTypeC dataType (T5SmallDecoderAttentionMask device dataType 1 decoderInputSeqSize),
    WithDeviceC device (WithDataTypeF dataType (Tensor 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)]))),
    WithDataTypeC dataType (Tensor 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])),
    WithDeviceC device (WithDataTypeF dataType (T5SmallDecoderAttentionMask device dataType 1 decoderInputSeqSize)),
    WithDataTypeC dataType (T5SmallDecoderAttentionMask device dataType 1 decoderInputSeqSize),
    KnownNat decoderInputSeqSize
  ) =>
  WithDeviceF
    device
    ( WithDataTypeF
        dataType
        (T5SmallDecoderAttentionMask device dataType 1 decoderInputSeqSize)
    )
t5SmallDecoderAttentionMask =
  withDevice @device $
    \deviceType ->
      withDataType
        @dataType
        @(T5SmallDecoderAttentionMask device dataType 1 decoderInputSeqSize)
        $ \dType ->
          let causalMask =
                unsqueeze @( 'SelectDim ( 'ByIndex 0))
                  . bool
                  . triu 1
                  $ withoutDataType
                    @dataType
                    @(Tensor 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)]))
                    ( withoutDevice @device
                        ( ones
                            @ 'WithoutGradient
                            @( 'Layout 'Dense)
                            @device
                            @dataType
                            @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])
                        )
                        deviceType
                    )
                    dType
           in maskedFill causalMask (-10000 :: Double) $
                withoutDataType
                  @dataType
                  @(T5SmallDecoderAttentionMask device dataType 1 decoderInputSeqSize)
                  ( withoutDevice @device
                      ( zeros
                          @ 'WithoutGradient
                          @( 'Layout 'Dense)
                          @device
                          @dataType
                          @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])
                      )
                      deviceType
                  )
                  dType

type T5SmallCrossAttentionMask device dataType batchSize inputSeqSize decoderInputSeqSize =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

t5SmallCrossAttentionMask ::
  forall device dataType inputSeqSize decoderInputSeqSize.
  ( WithDeviceC device (WithDataTypeF dataType (T5SmallCrossAttentionMask device dataType 1 inputSeqSize decoderInputSeqSize)),
    WithDataTypeC dataType (T5SmallCrossAttentionMask device dataType 1 inputSeqSize decoderInputSeqSize),
    KnownNat inputSeqSize,
    KnownNat decoderInputSeqSize
  ) =>
  WithDeviceF device (WithDataTypeF dataType (T5SmallCrossAttentionMask device dataType 1 inputSeqSize decoderInputSeqSize))
t5SmallCrossAttentionMask =
  zeros
    @ 'WithoutGradient
    @( 'Layout 'Dense)
    @device
    @dataType
    @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

type T5SmallInput device dataType batchSize inputSeqSize =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize), T5SmallInputEmbedDim])

type T5SmallDecoderInput device dataType batchSize decoderInputSeqSize =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), T5SmallInputEmbedDim])

type T5SmallDecoderOutput device dataType batchSize decoderInputSeqSize =
  Tensor
    'WithGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), T5SmallInputEmbedDim])

t5SmallFromPretrained :: FilePath -> Bool -> IO (T5Small ( 'Device 'CPU) ( 'DataType 'Float))
t5SmallFromPretrained filePath = runReaderT $ do
  iValue <- lift $ pickleLoad filePath
  stateDict <- case iValue of
    IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a state dictionary."
  T5Small
    <$> ( SequenceToSequenceTransformerWithLMHead
            <$> ( TransformerEncoder
                    <$> ( TransformerStackCons
                            <$> encoderBlock 0 stateDict
                            <*> ( TransformerStackCons
                                    <$> encoderBlock 1 stateDict
                                    <*> ( TransformerStackCons
                                            <$> encoderBlock 2 stateDict
                                            <*> ( TransformerStackCons
                                                    <$> encoderBlock 3 stateDict
                                                    <*> ( TransformerStackCons
                                                            <$> encoderBlock 4 stateDict
                                                            <*> ( TransformerStackCons
                                                                    <$> encoderBlock 5 stateDict
                                                                    <*> pure TransformerStackNil
                                                                )
                                                        )
                                                )
                                        )
                                )
                        )
                    <*> ( LayerNormWithoutBias
                            <$> lookup "encoder.final_layer_norm.weight" stateDict
                            <*> pure t5SmallEps
                        )
                    <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
                    <*> ( Embedding
                            <$> lookup "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" stateDict
                        )
                )
            <*> ( TransformerDecoder
                    <$> ( TransformerDecoderStackCons
                            <$> decoderBlock 0 stateDict
                            <*> ( TransformerDecoderStackCons
                                    <$> decoderBlock 1 stateDict
                                    <*> ( TransformerDecoderStackCons
                                            <$> decoderBlock 2 stateDict
                                            <*> ( TransformerDecoderStackCons
                                                    <$> decoderBlock 3 stateDict
                                                    <*> ( TransformerDecoderStackCons
                                                            <$> decoderBlock 4 stateDict
                                                            <*> ( TransformerDecoderStackCons
                                                                    <$> decoderBlock 5 stateDict
                                                                    <*> pure TransformerDecoderStackNil
                                                                )
                                                        )
                                                )
                                        )
                                )
                        )
                    <*> ( LayerNormWithoutBias
                            <$> lookup "decoder.final_layer_norm.weight" stateDict
                            <*> pure t5SmallEps
                        )
                    <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
                    <*> ( Embedding
                            <$> lookup "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" stateDict
                        )
                )
            <*> ( Embedding
                    <$> lookup "shared.weight" stateDict
                )
            <*> ( LinearWithoutBias
                    <$> lookup "lm_head.weight" stateDict
                )
        )
  where
    go [] = pure []
    go ((IVString s, IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((s, t) :) <$> go xs
    go ((_, IVTensor _) : _) = fail "iValue is not a string."
    go ((IVString _, _) : _) = fail "iValue is not a tensor."
    go _ = fail "iValue is neither a string nor a tensor."
    lookup s stateDict = do
      debug <- ask
      if debug
        then lift . putStrLn $ "loading `" <> s <> "`..."
        else pure ()
      lift
        ( maybe
            (fail $ "`" <> show s <> "` is not in the state dictionary.")
            (pure . UnsafeTensor)
            (Map.lookup s stateDict)
        )
        >>= checkedLayout
        >>= checkedDevice
        >>= checkedDataType
        >>= checkedShape
    headDim = case dimVal @T5SmallHeadDim of
      Dim (Name name) (Size size) -> pure $ Dim name size
      Dim _ _ -> lift $ fail "head dimension unspecified"
    headEmbedDim = case dimVal @T5SmallHeadEmbedDim of
      Dim (Name name) (Size size) -> pure $ Dim name size
      Dim _ _ -> lift $ fail "head embed dimension unspecified"
    encoderBlock n stateDict = do
      TransformerBlock
        <$> ( SelfAttention
                <$> ( MultiHeadAttention
                        <$> headDim
                        <*> headEmbedDim
                        <*> (LinearWithoutBias <$> lookup ("encoder.block." <> show n <> ".layer.0.SelfAttention.q.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("encoder.block." <> show n <> ".layer.0.SelfAttention.k.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("encoder.block." <> show n <> ".layer.0.SelfAttention.v.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("encoder.block." <> show n <> ".layer.0.SelfAttention.o.weight") stateDict)
                        <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookup ("encoder.block." <> show n <> ".layer.0.layer_norm.weight") stateDict
                        <*> pure t5SmallEps
                    )
                <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
            )
        <*> ( TransformerFeedForwardNetwork
                <$> (LinearWithoutBias <$> lookup ("encoder.block." <> show n <> ".layer.1.DenseReluDense.wi.weight") stateDict)
                <*> (LinearWithoutBias <$> lookup ("encoder.block." <> show n <> ".layer.1.DenseReluDense.wo.weight") stateDict)
                <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
                <*> ( LayerNormWithoutBias
                        <$> lookup ("encoder.block." <> show n <> ".layer.1.layer_norm.weight") stateDict
                        <*> pure t5SmallEps
                    )
                <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
            )
    decoderBlock n stateDict =
      TransformerDecoderBlock
        <$> ( SelfAttention
                <$> ( MultiHeadAttention
                        <$> headDim
                        <*> headEmbedDim
                        <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.0.SelfAttention.q.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.0.SelfAttention.k.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.0.SelfAttention.v.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.0.SelfAttention.o.weight") stateDict)
                        <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookup ("decoder.block." <> show n <> ".layer.0.layer_norm.weight") stateDict
                        <*> pure t5SmallEps
                    )
                <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
            )
        <*> ( CrossAttention
                <$> ( MultiHeadAttention
                        <$> headDim
                        <*> headEmbedDim
                        <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.1.EncDecAttention.q.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.1.EncDecAttention.k.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.1.EncDecAttention.v.weight") stateDict)
                        <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.1.EncDecAttention.o.weight") stateDict)
                        <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookup ("decoder.block." <> show n <> ".layer.1.layer_norm.weight") stateDict
                        <*> pure t5SmallEps
                    )
                <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
            )
        <*> ( TransformerFeedForwardNetwork
                <$> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.2.DenseReluDense.wi.weight") stateDict)
                <*> (LinearWithoutBias <$> lookup ("decoder.block." <> show n <> ".layer.2.DenseReluDense.wo.weight") stateDict)
                <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
                <*> ( LayerNormWithoutBias
                        <$> lookup ("decoder.block." <> show n <> ".layer.2.layer_norm.weight") stateDict
                        <*> pure t5SmallEps
                    )
                <*> pure (initialize @(Dropout Float) t5SmallDropoutP)
            )

-- forwardT5Small ::
--   forall device dataType (batchSize :: Nat) (inputSeqSize :: Nat) (decoderInputSeqSize :: Nat) .
--   ( KnownNat batchSize,
--     KnownNat inputSeqSize,
--     KnownNat decoderInputSeqSize,
--     NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size inputSeqSize), T5SmallEmbedDim]) ~ NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size inputSeqSize), T5SmallHeadDim, T5SmallHeadEmbedDim]),
--     NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size decoderInputSeqSize), T5SmallEmbedDim]) ~ NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size decoderInputSeqSize), T5SmallHeadDim, T5SmallHeadEmbedDim])
--   ) =>
--   T5Small device dataType ->
--   T5SmallInput device dataType batchSize inputSeqSize ->
--   T5SmallDecoderInput device dataType batchSize decoderInputSeqSize ->
--   T5SmallAttentionMask device dataType batchSize inputSeqSize ->
--   T5SmallDecoderAttentionMask device dataType batchSize decoderInputSeqSize ->
--   T5SmallCrossAttentionMask device dataType batchSize inputSeqSize decoderInputSeqSize ->
--   Generator device ->
--   ( T5SmallDecoderOutput device dataType batchSize decoderInputSeqSize
--   , Generator device
--   )
-- forwardT5Small model input decoderInput attentionMask decoderAttentionMask crossAttentionMask =
--   forward model (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)

-- >>> relPos 32 128 21 17
-- [[0,17,18,19,20,21,22,23,24,24,24,24,25,25,25,25,26],[1,0,17,18,19,20,21,22,23,24,24,24,24,25,25,25,25],[2,1,0,17,18,19,20,21,22,23,24,24,24,24,25,25,25],[3,2,1,0,17,18,19,20,21,22,23,24,24,24,24,25,25],[4,3,2,1,0,17,18,19,20,21,22,23,24,24,24,24,25],[5,4,3,2,1,0,17,18,19,20,21,22,23,24,24,24,24],[6,5,4,3,2,1,0,17,18,19,20,21,22,23,24,24,24],[7,6,5,4,3,2,1,0,17,18,19,20,21,22,23,24,24],[8,7,6,5,4,3,2,1,0,17,18,19,20,21,22,23,24],[8,8,7,6,5,4,3,2,1,0,17,18,19,20,21,22,23],[8,8,8,7,6,5,4,3,2,1,0,17,18,19,20,21,22],[8,8,8,8,7,6,5,4,3,2,1,0,17,18,19,20,21],[9,8,8,8,8,7,6,5,4,3,2,1,0,17,18,19,20],[9,9,8,8,8,8,7,6,5,4,3,2,1,0,17,18,19],[9,9,9,8,8,8,8,7,6,5,4,3,2,1,0,17,18],[9,9,9,9,8,8,8,8,7,6,5,4,3,2,1,0,17],[10,9,9,9,9,8,8,8,8,7,6,5,4,3,2,1,0],[10,10,9,9,9,9,8,8,8,8,7,6,5,4,3,2,1],[10,10,10,9,9,9,9,8,8,8,8,7,6,5,4,3,2],[10,10,10,10,9,9,9,9,8,8,8,8,7,6,5,4,3],[10,10,10,10,10,9,9,9,9,8,8,8,8,7,6,5,4]]
relPos :: Int -> Int -> Int -> Int -> [[Int]]
relPos numBuckets maxDistance querySize keySize =
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

-- >>> decoderRelPos 32 128 21 17
-- [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0],[6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0],[7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0],[8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0],[9,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0],[10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0],[11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0],[12,11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0],[13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,0,0],[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,0],[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0],[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],[16,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1],[16,16,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2],[17,16,16,16,15,14,13,12,11,10,9,8,7,6,5,4,3],[17,17,16,16,16,15,14,13,12,11,10,9,8,7,6,5,4]]
decoderRelPos :: Int -> Int -> Int -> Int -> [[Int]]
decoderRelPos numBuckets maxDistance querySize keySize =
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

testForwardT5Small :: IO ()
testForwardT5Small =
  let attentionMask = t5SmallAttentionMask @( 'Device 'CPU) @( 'DataType 'Float) @15
      decoderAttentionMask = t5SmallDecoderAttentionMask @( 'Device 'CPU) @( 'DataType 'Float) @4
      crossAttentionMask = t5SmallCrossAttentionMask @( 'Device 'CPU) @( 'DataType 'Float) @15 @4
   in do
        input <-
          case Torch.Tensor.asTensor [[6536 :: Int, 43, 118, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1]] of
            Torch.Tensor.Unsafe t ->
              pure (UnsafeTensor @ 'WithoutGradient t)
                >>= checkedLayout @( 'Layout 'Dense)
                >>= checkedDevice @( 'Device 'CPU)
                >>= checkedDataType @( 'DataType 'Int64)
                >>= checkedShape @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size 15)])
        decoderInput <-
          case Torch.Tensor.asTensor [[6536 :: Int, 504, 24, 1]] of
            Torch.Tensor.Unsafe t ->
              pure (UnsafeTensor @ 'WithoutGradient t)
                >>= checkedLayout @( 'Layout 'Dense)
                >>= checkedDevice @( 'Device 'CPU)
                >>= checkedDataType @( 'DataType 'Int64)
                >>= checkedShape @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size 4)])
        relPos <-
          case Torch.Tensor.asTensor [relPos 32 128 15 15] of
            Torch.Tensor.Unsafe t ->
              pure (UnsafeTensor @ 'WithoutGradient t)
                >>= checkedLayout @( 'Layout 'Dense)
                >>= checkedDevice @( 'Device 'CPU)
                >>= checkedDataType @( 'DataType 'Int64)
                >>= checkedShape @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size 15), 'Dim ( 'Name "*") ( 'Size 15)])
        decoderRelPos <-
          case Torch.Tensor.asTensor [decoderRelPos 32 128 4 4] of
            Torch.Tensor.Unsafe t ->
              pure (UnsafeTensor @ 'WithoutGradient t)
                >>= checkedLayout @( 'Layout 'Dense)
                >>= checkedDevice @( 'Device 'CPU)
                >>= checkedDataType @( 'DataType 'Int64)
                >>= checkedShape @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size 4), 'Dim ( 'Name "*") ( 'Size 4)])
        model <- t5SmallFromPretrained "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt" False
        g <- mkGenerator @( 'Device CPU) 0
        -- let (output, _) = forward model (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask) g
        -- case output of
        --   UnsafeTensor t -> print . Torch.Tensor.Unsafe $ t
        pure ()
  where
    ones' ::
      forall shape.
      WithShapeC shape (Tensor 'WithoutGradient ( 'Layout 'Dense) ( 'Device 'CPU) ( 'DataType 'Float) shape) =>
      WithShapeF shape (Tensor 'WithoutGradient ( 'Layout 'Dense) ( 'Device 'CPU) ( 'DataType 'Float) shape)
    ones' = ones @ 'WithoutGradient @( 'Layout 'Dense) @( 'Device 'CPU) @( 'DataType 'Float) @shape

-- forwardT5Small' ::
--   forall device dataType .
--   T5Small device dataType ->
--   Tensor 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, T5SmallInputEmbedDim]) ->
--   Tensor 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, T5SmallDecoderInputEmbedDim]) ->
--   Tensor 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Tensor 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Tensor 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Generator device ->
--   ( Tensor 'WithGradient ( 'Layout 'Dense) device dataType 'UncheckedShape
--   , Generator device
--   )
-- forwardT5Small' (T5Small seqToSeq) input decoderInput attentionMask decoderAttentionMask crossAttentionMask =
--   forward seqToSeq (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)

-- forwardT5Small'' ::
--   T5Small 'UncheckedDevice 'UncheckedDataType ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, T5SmallInputEmbedDim]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize, T5SmallDecoderInputEmbedDim]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType ( 'Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]) ->
--   Generator 'UncheckedDevice ->
--   ( Tensor 'Dependent ( 'Layout 'Dense) 'UncheckedDevice 'UncheckedDataType 'UncheckedShape
--   , Generator 'UncheckedDevice
--   )
-- forwardT5Small'' model input decoderInput attentionMask decoderAttentionMask crossAttentionMask =
--   forward model (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)

-- forwardT5Small''' ::
--   forall (batchSize :: Nat) (inputSeqSize :: Nat) (decoderInputSeqSize :: Nat) .
--   ( KnownNat batchSize,
--     KnownNat inputSeqSize,
--     KnownNat decoderInputSeqSize,
--     NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size inputSeqSize), T5SmallEmbedDim]) ~ NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size inputSeqSize), T5SmallHeadDim, T5SmallHeadEmbedDim]),
--     NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size decoderInputSeqSize), T5SmallEmbedDim]) ~ NumelF ('Shape '[ 'Dim ('Name "*") ('Size batchSize), 'Dim ('Name "*") ('Size decoderInputSeqSize), T5SmallHeadDim, T5SmallHeadEmbedDim])
--   ) =>
--   T5Small 'UncheckedDevice 'UncheckedDataType ->
--   T5SmallInput 'UncheckedDevice 'UncheckedDataType batchSize inputSeqSize ->
--   T5SmallDecoderInput 'UncheckedDevice 'UncheckedDataType batchSize decoderInputSeqSize ->
--   T5SmallAttentionMask 'UncheckedDevice 'UncheckedDataType inputSeqSize ->
--   T5SmallDecoderAttentionMask 'UncheckedDevice 'UncheckedDataType decoderInputSeqSize ->
--   T5SmallCrossAttentionMask 'UncheckedDevice 'UncheckedDataType inputSeqSize decoderInputSeqSize ->
--   Generator 'UncheckedDevice ->
--   ( T5SmallDecoderOutput 'UncheckedDevice 'UncheckedDataType batchSize decoderInputSeqSize
--   , Generator 'UncheckedDevice
--   )
-- forwardT5Small''' model input decoderInput attentionMask decoderAttentionMask crossAttentionMask =
--   forward model (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)
