{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL #-}

module Torch.GraduallyTyped.NN.Transformer.Encoder where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Constraint, Type)
import Data.Singletons (SingI, sing)
import GHC.TypeLits (Nat, Symbol, type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF)
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), HasInitializeEmbeddingC)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasInitializeTransformerStack, HasInitializeTransformerStackC, HasLookupStack, TransformerStack, lookupStack)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim, Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithDimsC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (Unify, type (<+>))

-- | Generic transformer encoder.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'TransformerEncoder'.
data
  GTransformerEncoder
    (stack :: Type)
    (embedLayerNorm :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
    (posEnc :: Type)
  where
  GTransformerEncoder ::
    forall stack embedLayerNorm layerNorm dropout posEnc.
    { -- | encoder layer stack
      teStack :: stack,
      -- | encoder embedding layer norm
      teEmbedLayerNorm :: embedLayerNorm,
      -- | encoder layer norm
      teLayerNorm :: layerNorm,
      -- | encoder dropout
      teDropout :: dropout,
      -- | positional encoding
      tePosEnc :: posEnc
    } ->
    GTransformerEncoder stack embedLayerNorm layerNorm dropout posEnc

-- | Transformer encoder.
newtype
  TransformerEncoder
    (numLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerEncoder ::
    forall numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP.
    GTransformerEncoderF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP ->
    TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP

type GTransformerEncoderF
  (numLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GTransformerEncoder
    (TEStackF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
    (TEEmbedLayerNormF style device dataType inputEmbedDim)
    (TELayerNormF style device dataType inputEmbedDim)
    (TEDropoutF style dropoutP)
    (TEPosEncF style device dataType headDim inputEmbedDim posEncDim)

type family
  TEStackF
    (numLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  TEStackF numLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP =
    TransformerStack numLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP

type family
  TEEmbedLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEEmbedLayerNormF 'T5 _ _ _ = ()
  TEEmbedLayerNormF 'BERT device dataType inputEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[inputEmbedDim])
  TEEmbedLayerNormF 'Pegasus _ _ _ = ()

type family
  TELayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TELayerNormF 'T5 device dataType inputEmbedDim =
    LayerNorm 'WithoutBias device dataType ('Shape '[inputEmbedDim])
  TELayerNormF 'BERT _ _ _ = ()
  TELayerNormF 'Pegasus device dataType inputEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[inputEmbedDim])

type family
  TEDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  TEDropoutF _ dropoutP = Dropout dropoutP

type family
  TEPosEncF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEPosEncF 'T5 device dataType headDim _ posEncDim =
    Embedding ('Layout 'Dense) device dataType posEncDim headDim 'Nothing
  TEPosEncF 'BERT device dataType _ inputEmbedDim posEncDim =
    Embedding ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing
  TEPosEncF 'Pegasus device dataType _ inputEmbedDim posEncDim =
    Embedding ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing

type HasInitializeTransformerEncoderC
  (numLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device))))),
    WithDimC inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device)))),
    WithDimC ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device))),
    WithDimC posEncDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device))
  )

type family
  HasInitializeTEEmbedLayerNormF
    (embedLayerNorm :: Type)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Constraint
  where
  HasInitializeTEEmbedLayerNormF _ 'T5 _ _ _ = ()
  HasInitializeTEEmbedLayerNormF embedLayerNorm 'BERT device dataType inputEmbedDim =
    ( HasInitialize embedLayerNorm,
      InitializeF embedLayerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[inputEmbedDim] (Double -> embedLayerNorm))),
      WithDeviceC device (WithDataTypeF dataType (WithDimsF '[inputEmbedDim] (Double -> embedLayerNorm))),
      WithDataTypeC dataType (WithDimsF '[inputEmbedDim] (Double -> embedLayerNorm)),
      WithDimsC '[inputEmbedDim] (Double -> embedLayerNorm)
    )
  HasInitializeTEEmbedLayerNormF _ 'Pegasus _ _ _ = ()

type family
  HasInitializeTELayerNormF
    (layerNorm :: Type)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Constraint
  where
  HasInitializeTELayerNormF layerNorm 'T5 device dataType inputEmbedDim =
    ( HasInitialize layerNorm,
      InitializeF layerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[inputEmbedDim] (Double -> layerNorm))),
      WithDeviceC device (WithDataTypeF dataType (WithDimsF '[inputEmbedDim] (Double -> layerNorm))),
      WithDataTypeC dataType (WithDimsF '[inputEmbedDim] (Double -> layerNorm)),
      WithDimsC '[inputEmbedDim] (Double -> layerNorm)
    )
  HasInitializeTELayerNormF _ 'BERT _ _ _ = ()
  HasInitializeTELayerNormF layerNorm 'Pegasus device dataType inputEmbedDim =
    ( HasInitialize layerNorm,
      InitializeF layerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[inputEmbedDim] (Double -> layerNorm))),
      WithDeviceC device (WithDataTypeF dataType (WithDimsF '[inputEmbedDim] (Double -> layerNorm))),
      WithDataTypeC dataType (WithDimsF '[inputEmbedDim] (Double -> layerNorm)),
      WithDimsC '[inputEmbedDim] (Double -> layerNorm)
    )

type family
  HasInitializeTEPosEncF
    (posEnc :: Type)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Constraint
  where
  HasInitializeTEPosEncF posEnc 'T5 device dataType headDim _ posEncDim =
    ( HasInitialize posEnc,
      InitializeF posEnc ~ WithDeviceF device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF headDim (Generator device -> (posEnc, Generator device))))),
      WithDeviceC device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF headDim (Generator device -> (posEnc, Generator device))))),
      WithDataTypeC dataType (WithDimF posEncDim (WithDimF headDim (Generator device -> (posEnc, Generator device)))),
      WithDimC posEncDim (WithDimF headDim (Generator device -> (posEnc, Generator device))),
      WithDimC headDim (Generator device -> (posEnc, Generator device))
    )
  HasInitializeTEPosEncF posEnc 'BERT device dataType _ inputEmbedDim posEncDim =
    ( HasInitialize posEnc,
      InitializeF posEnc ~ WithDeviceF device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF inputEmbedDim (Generator device -> (posEnc, Generator device))))),
      WithDeviceC device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF inputEmbedDim (Generator device -> (posEnc, Generator device))))),
      WithDataTypeC dataType (WithDimF posEncDim (WithDimF inputEmbedDim (Generator device -> (posEnc, Generator device)))),
      WithDimC posEncDim (WithDimF inputEmbedDim (Generator device -> (posEnc, Generator device))),
      WithDimC inputEmbedDim (Generator device -> (posEnc, Generator device))
    )
  HasInitializeTEPosEncF posEnc 'Pegasus device dataType _ inputEmbedDim posEncDim =
    ( HasInitialize posEnc,
      InitializeF posEnc ~ WithDeviceF device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF inputEmbedDim (Generator device -> (posEnc, Generator device))))),
      WithDeviceC device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF inputEmbedDim (Generator device -> (posEnc, Generator device))))),
      WithDataTypeC dataType (WithDimF posEncDim (WithDimF inputEmbedDim (Generator device -> (posEnc, Generator device)))),
      WithDimC posEncDim (WithDimF inputEmbedDim (Generator device -> (posEnc, Generator device))),
      WithDimC inputEmbedDim (Generator device -> (posEnc, Generator device))
    )

instance
  ( SingI style,
    HasInitializeTransformerEncoderC numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP,
    stack ~ TEStackF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP,
    HasInitialize stack,
    InitializeF stack ~ WithDeviceF device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device)))))))),
    HasInitializeTransformerStackC stack device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP,
    embedLayerNorm ~ TEEmbedLayerNormF style device dataType inputEmbedDim,
    HasInitializeTEEmbedLayerNormF embedLayerNorm style device dataType inputEmbedDim,
    layerNorm ~ TELayerNormF style device dataType inputEmbedDim,
    HasInitializeTELayerNormF layerNorm style device dataType inputEmbedDim,
    dropout ~ TEDropoutF style dropoutP,
    HasInitialize dropout,
    InitializeF dropout ~ (dropoutP -> dropout),
    posEnc ~ TEPosEncF style device dataType headDim inputEmbedDim posEncDim,
    HasInitializeTEPosEncF posEnc style device dataType headDim inputEmbedDim posEncDim
  ) =>
  HasInitialize (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
  where
  type
    InitializeF (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                headDim
                ( WithDimF
                    headEmbedDim
                    ( WithDimF
                        embedDim
                        ( WithDimF
                            inputEmbedDim
                            ( WithDimF
                                ffnDim
                                ( WithDimF
                                    posEncDim
                                    (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device))
                                )
                            )
                        )
                    )
                )
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @headDim $
              \headDim ->
                withDim @headEmbedDim $
                  \headEmbedDim ->
                    withDim @embedDim $
                      \embedDim ->
                        withDim @inputEmbedDim $
                          \inputEmbedDim ->
                            withDim @ffnDim $
                              \ffnDim ->
                                withDim @posEncDim @(dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP, Generator device)) $
                                  \posEncDim -> go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim
    where
      go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps = runState $ do
        stack <-
          state $
            withoutDim @ffnDim @(dropoutP -> Double -> Generator device -> (stack, Generator device))
              ( withoutDim @inputEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @stack
                                      )
                                      deviceType
                                  )
                                  dType
                              )
                              headDim
                          )
                          headEmbedDim
                      )
                      embedDim
                  )
                  inputEmbedDim
              )
              ffnDim
              dropoutP
              eps
        let embedLayerNorm = case sing @style of
              ST5 -> ()
              SBERT ->
                withoutShape @('Shape '[inputEmbedDim]) @(Double -> embedLayerNorm)
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @embedLayerNorm
                          )
                          deviceType
                      )
                      dType
                  )
                  [inputEmbedDim]
                  eps
              SPegasus -> ()
        let layerNorm = case sing @style of
              ST5 ->
                withoutShape @('Shape '[inputEmbedDim]) @(Double -> layerNorm)
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @layerNorm
                          )
                          deviceType
                      )
                      dType
                  )
                  [inputEmbedDim]
                  eps
              SBERT -> ()
              SPegasus ->
                withoutShape @('Shape '[inputEmbedDim]) @(Double -> layerNorm)
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @layerNorm
                          )
                          deviceType
                      )
                      dType
                  )
                  [inputEmbedDim]
                  eps
        let dropout = initialize @dropout dropoutP
        posEnc <-
          state $ case sing @style of
            ST5 ->
              withoutDim @headDim @(Generator device -> (posEnc, Generator device))
                ( withoutDim @posEncDim
                    ( withoutDataType @dataType
                        ( withoutDevice @device
                            ( initialize @posEnc
                            )
                            deviceType
                        )
                        dType
                    )
                    posEncDim
                )
                headDim
            SBERT ->
              withoutDim @inputEmbedDim @(Generator device -> (posEnc, Generator device))
                ( withoutDim @posEncDim
                    ( withoutDataType @dataType
                        ( withoutDevice @device
                            ( initialize @posEnc
                            )
                            deviceType
                        )
                        dType
                    )
                    posEncDim
                )
                inputEmbedDim
            SPegasus ->
              withoutDim @inputEmbedDim @(Generator device -> (posEnc, Generator device))
                ( withoutDim @posEncDim
                    ( withoutDataType @dataType
                        ( withoutDevice @device
                            ( initialize @posEnc
                            )
                            deviceType
                        )
                        dType
                    )
                    posEncDim
                )
                inputEmbedDim
        pure . TransformerEncoder $ GTransformerEncoder stack embedLayerNorm layerNorm dropout posEnc

lookupEncoder ::
  forall numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim inputEmbedDim,
    KnownDim ffnDim,
    KnownDim posEncDim,
    Scalar dropoutP,
    HasLookupStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP m
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
lookupEncoder dropoutP eps prefix =
  let stack ST5 = lookupStack dropoutP eps (prefix <> "block.")
      embedLayerNorm ST5 = pure @m ()
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "final_layer_norm.weight")
          <*> pure eps
      dropout ST5 = pure (initialize @(Dropout dropoutP) dropoutP)
      posEnc ST5 = fmap @m Embedding $ lookupTensor (prefix <> "block.0.layer.0.SelfAttention.relative_attention_bias.weight")
   in TransformerEncoder
        <$> ( GTransformerEncoder
                <$> stack (sing @style)
                <*> embedLayerNorm (sing @style)
                <*> layerNorm (sing @style)
                <*> dropout (sing @style)
                <*> posEnc (sing @style)
            )

-- | 'HasForward' instance for @TransformerEncoder numLayers 'T5@.
--
-- @
--  ┌───────┐  ┌────────┐  ┌───────────────┐
--  │ input │  │ relPos │  │ attentionMask │
--  └───┬───┘  └───┬────┘  └───────┬───────┘
--      │          │               │
--      │          ▼               │
--      │      tePosEnc            │
--      │          ▼               │
--      │      transpose           │
--      │          ▼               ▼
--      │      transpose       unsqueeze
--      ▼          │               │
--  teDropout      └─────►add◄─────┘
--      ▼                  │
--   teStack◄──────────────┘
--      ▼
-- teLayerNorm
--      ▼
--  teDropout
--      │
--      ▼
--  ┌────────┐
--  │ output │
--  └────────┘
-- @
instance
  ( HasForward
      (TEDropoutF 'T5 dropoutP)
      input
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          'WithGradient
          ('Layout 'Dense <+> relPosLayout <+> attentionMaskLayout)
          (device <+> relPosDevice <+> attentionMaskDevice)
          (Seq (relPosDataType <+> 'DataType 'Int64) dataType <+> attentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      ( EmbeddingF
                          ('Shape '[posEncDim, headDim])
                          relPosShape
                      )
                  )
              )
              ( UnsqueezeF
                  ('SelectDim ('ByIndex 1))
                  attentionMaskShape
              )
          )
      )
      dropoutGeneratorOutput
      stackOutput
      stackGeneratorOutput,
    HasForward
      (TELayerNormF 'T5 device dataType inputEmbedDim)
      stackOutput
      stackGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (TEDropoutF 'T5 dropoutP)
      layerNormOutput
      layerNormGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerEncoder numLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( input,
      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, relPos, attentionMask) =
    let relPosBias =
          ireturn relPos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        attentionBias =
          relPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) attentionMask)
     in runIxState $
          ireturn input
            >>>= IxState . forward teDropout
            >>>= (\input' -> attentionBias >>>= (\attentionBias' -> IxState $ forward teStack (input', attentionBias')))
            >>>= IxState . forward teLayerNorm
            >>>= IxState . forward teDropout

-- | 'HasForward' instance for @TransformerEncoder numLayers 'BART@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  │
--   teEmbedLayerNorm          │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @

-- | 'HasForward' instance for @TransformerEncoder numLayers 'MBART@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  │
--   teEmbedLayerNorm          │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          ▼
--     teLayerNorm
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @

-- | 'HasForward' instance for @TransformerEncoder numLayers 'BERT@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  │
--   teEmbedLayerNorm          │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      (TEEmbedLayerNormF 'BERT device dataType inputEmbedDim)
      ( Tensor
          'WithGradient
          (inputLayout <+> 'Layout 'Dense <+> posLayout)
          (inputDevice <+> device <+> posDevice)
          (inputDataType <+> Seq (posDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF inputShape (EmbeddingF ('Shape '[posEncDim, inputEmbedDim]) posShape))
      )
      generator
      layerNormOutput
      generator,
    HasForward
      (TEDropoutF 'BERT dropoutP)
      layerNormOutput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'BERT device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskRequiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerEncoder numLayers 'BERT device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posRequiresGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxState $
          ireturn pos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxState . forward teEmbedLayerNorm
            >>>= IxState . forward teDropout
            >>>= (\input' -> IxState $ forward teStack (input', attentionBias))

-- | 'HasForward' instance for @TransformerEncoder numLayers 'Pegasus@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          ▼
--     teLayerNorm
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      (TEDropoutF 'Pegasus dropoutP)
      ( Tensor
          'WithGradient
          (inputLayout <+> 'Layout 'Dense <+> posLayout)
          (inputDevice <+> device <+> posDevice)
          (inputDataType <+> Seq (posDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF inputShape (EmbeddingF ('Shape '[posEncDim, inputEmbedDim]) posShape))
      )
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'Pegasus device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskRequiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutput
      stackOutput
      generatorOutput,
    HasForward
      (TELayerNormF 'Pegasus device dataType inputEmbedDim)
      stackOutput
      generatorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerEncoder numLayers 'Pegasus device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posRequiresGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxState $
          ireturn pos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxState . forward teDropout
            >>>= (\input' -> IxState $ forward teStack (input', attentionBias))
            >>>= IxState . forward teLayerNorm