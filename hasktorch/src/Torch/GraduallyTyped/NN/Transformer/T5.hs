{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
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

import Control.Monad.Reader (ReaderT (runReaderT), ask, lift)
import qualified Data.Map as Map
import GHC.TypeLits (KnownNat, Nat)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Linear (Linear (LinearWithoutBias))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.Block (TransformerBlock (TransformerBlock))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention (CrossAttention))
import Torch.GraduallyTyped.NN.Transformer.Decoder (TransformerDecoder (TransformerDecoder))
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (TransformerDecoderBlock (TransformerDecoderBlock))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (TransformerDecoderStack (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder (..))
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork (TransformerFeedForwardNetwork))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (MultiHeadAttention (MultiHeadAttention))
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention (SelfAttention))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer (..))
import Torch.GraduallyTyped.NN.Transformer.Stack (TransformerStack (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (NumelDimF, NumelF, Shape (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), Name (..), SelectDim (..), Size (..))
import Torch.GraduallyTyped.Tensor.Creation (ones, zeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (UnsqueezeF, unsqueeze)
import Torch.GraduallyTyped.Tensor.Other (maskedFill, triu)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), bool, checkedDataType, checkedDevice, checkedLayout, checkedShape, shape)
import Torch.Script (IValue (..))
import Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe))
import Torch.GraduallyTyped.Random (mkGenerator)

-- | num_layers = 6
type T5SmallNumLayers = 6

-- | n_heads = 8
type T5SmallHeadDim = 'Dim ( 'Name "head") ( 'Size 8)

-- | d_kv = 64
type T5SmallHeadEmbedDim = 'Dim ( 'Name "headEmbed") ( 'Size 64)

-- | inner_dim =  = n_heads * d_kv = 512
type T5SmallEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | d_model = 512
type T5SmallInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | d_model = 512
type T5SmallDecoderInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | d_ff = 2048
type T5SmallFFNDim = 'Dim ( 'Name "*") ( 'Size 2048)

-- | https://huggingface.co/t5-small/blob/main/config.json
data T5Small device dataType where
  T5Small ::
    forall device dataType.
    SequenceToSequenceTransformer
      T5SmallNumLayers
      T5SmallNumLayers
      device
      dataType
      T5SmallHeadDim
      T5SmallHeadEmbedDim
      T5SmallEmbedDim
      T5SmallInputEmbedDim
      T5SmallDecoderInputEmbedDim
      T5SmallFFNDim
      Float ->
    T5Small device dataType

instance
  HasForward
    (SequenceToSequenceTransformer T5SmallNumLayers T5SmallNumLayers device dataType T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallDecoderInputEmbedDim T5SmallFFNDim Float)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    (Generator generatorDevice) =>
  HasForward
    (T5Small device dataType)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
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
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardOutput
        (SequenceToSequenceTransformer T5SmallNumLayers T5SmallNumLayers device dataType T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallDecoderInputEmbedDim T5SmallFFNDim Float)
        ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
          Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
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
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardGeneratorOutput
        (SequenceToSequenceTransformer T5SmallNumLayers T5SmallNumLayers device dataType T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallDecoderInputEmbedDim T5SmallFFNDim Float)
        ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
          Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
          Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
          Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        (Generator generatorDevice)
  -- forward (T5Small seqToSeq) inputs = forward seqToSeq inputs

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
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), T5SmallDecoderInputEmbedDim])

type T5SmallDecoderOutput device dataType batchSize decoderInputSeqSize =
  Tensor
    'WithGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), T5SmallDecoderInputEmbedDim])

t5SmallFromPretrained :: FilePath -> Bool -> IO (T5Small ( 'Device 'CPU) ( 'DataType 'Float))
t5SmallFromPretrained filePath = runReaderT $ do
  iValue <- lift $ pickleLoad filePath
  stateDict <- case iValue of
    IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a state dictionary."
  seqToSeq <- do
    encoder <- do
      stack <-
        TransformerStackCons
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
      layerNorm <-
        LayerNormWithoutBias
          <$> lookup "encoder.final_layer_norm.weight" stateDict
          <*> pure t5SmallEps
      let dropout = initialize @(Dropout Float) t5SmallDropoutP
      pure $ TransformerEncoder stack layerNorm dropout
    decoder <- do
      stack <-
        TransformerDecoderStackCons
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
      layerNorm <-
        LayerNormWithoutBias
          <$> lookup "decoder.final_layer_norm.weight" stateDict
          <*> pure t5SmallEps
      let dropout = initialize @(Dropout Float) t5SmallDropoutP
      pure $ TransformerDecoder stack layerNorm dropout
    pure $ SequenceToSequenceTransformer encoder decoder
  pure $ T5Small seqToSeq
  where
    go [] = pure []
    go ((IVString s, IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((s, t) :) <$> go xs
    go ((_, IVTensor _) : _) = fail "iValue is not a string."
    go ((IVString _, _) : _) = fail "iValue is not a tensor."
    go _ = fail "iValue is neither a string nor a tensor."
    lookup s stateDict = do
      debug <- ask
      if debug
        then lift . putStrLn $ "loading `" <> s <> "`."
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

-- testForwardT5Small :: IO ()
-- testForwardT5Small =
--   let input = ones @'WithoutGradient @( 'Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size 3), T5SmallInputEmbedDim])
--       decoderInput = ones @'WithoutGradient @( 'Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size 2), T5SmallDecoderInputEmbedDim])
--       attentionMask = t5SmallAttentionMask @('Device 'CPU) @('DataType 'Float) @3
--       decoderAttentionMask = t5SmallDecoderAttentionMask @('Device 'CPU) @('DataType 'Float) @2
--       crossAttentionMask = t5SmallCrossAttentionMask @('Device 'CPU) @('DataType 'Float) @3 @2
--    in do
--      model <- t5SmallFromPretrained "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt" False
--      g <- mkGenerator @('Device CPU) 0
--      let (output, _) = forward model (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask) g
--      case output of
--        UnsafeTensor t -> print . Torch.Tensor.Unsafe $ t
--      pure ()

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
