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

module Torch.GraduallyTyped.NN.Transformer.Decoder where

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
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), HasInitializeEmbeddingC)
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasInitializeTransformerDecoderStack, HasInitializeTransformerDecoderStackC, HasLookupDecoderStack, TransformerDecoderStack, lookupDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), KnownDim, Name (..), Shape (..), Size (..), WithDimC (..), WithDimsC (..), WithShapeC (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), SelectDim (..))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Generic transformer decoder.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'TransformerDecoder'.
data
  GTransformerDecoder
    (stack :: Type)
    (embedLayerNorm :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
    (posEnc :: Type)
  where
  GTransformerDecoder ::
    forall stack embedLayerNorm layerNorm dropout posEnc.
    { -- | decoder layer stack
      tdStack :: stack,
      -- | decoder embedding layer norm
      tdEmbedLayerNorm :: embedLayerNorm,
      -- | decoder layer norm
      tdLayerNorm :: layerNorm,
      -- | decoder dropout
      tdDropout :: dropout,
      -- | positional encoding
      tdPosEnc :: posEnc
    } ->
    GTransformerDecoder stack embedLayerNorm layerNorm dropout posEnc

-- | Transformer decoder.
data
  TransformerDecoder
    (numLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerDecoder ::
    forall numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP.
    GTransformerDecoderF numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP ->
    TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP

type GTransformerDecoderF
  (numLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GTransformerDecoder
    (TDStackF numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
    (TDEmbedLayerNormF style device dataType decoderInputEmbedDim)
    (TDLayerNormF style device dataType decoderInputEmbedDim)
    (TDDropoutF style dropoutP)
    (TDPosEncF style device dataType headDim decoderInputEmbedDim posEncDim)

type family
  TDStackF
    (numLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  TDStackF numLayers 'T5 device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP =
    TransformerDecoderStack numLayers 'T5 device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP

type family
  TDEmbedLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDEmbedLayerNormF 'T5 _ _ _ = ()
  TDEmbedLayerNormF 'BERT device dataType decoderInputEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[decoderInputEmbedDim])
  TDEmbedLayerNormF 'Pegasus _ _ _ = ()

type family
  TDLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDLayerNormF 'T5 device dataType decoderInputEmbedDim =
    LayerNorm 'WithoutBias device dataType ('Shape '[decoderInputEmbedDim])
  TDLayerNormF 'BERT _ _ _ = ()
  TDLayerNormF 'Pegasus device dataType decoderInputEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[decoderInputEmbedDim])

type family
  TDDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  TDDropoutF _ dropoutP = Dropout dropoutP

type family
  TDPosEncF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDPosEncF 'T5 device dataType headDim _ posEncDim =
    Embedding ('Layout 'Dense) device dataType posEncDim headDim 'Nothing
  TDPosEncF 'BERT device dataType _ decoderInputEmbedDim posEncDim =
    Embedding ('Layout 'Dense) device dataType posEncDim decoderInputEmbedDim 'Nothing
  TDPosEncF 'Pegasus device dataType _ decoderInputEmbedDim posEncDim =
    Embedding ('Layout 'Dense) device dataType posEncDim decoderInputEmbedDim 'Nothing

type HasInitializeTransformerDecoderC
  (numLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device)))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device))))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device)))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device))))))),
    WithDimC embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device)))))),
    WithDimC decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device))))),
    WithDimC encoderOutputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device)))),
    WithDimC ffnDim (WithDimF posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device))),
    WithDimC posEncDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device))
  )

type family
  HasInitializeTDEmbedLayerNormF
    (embedLayerNorm :: Type)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Constraint
  where
  HasInitializeTDEmbedLayerNormF _ 'T5 _ _ _ = ()
  HasInitializeTDEmbedLayerNormF embedLayerNorm 'BERT device dataType decoderInputEmbedDim =
    ( HasInitialize embedLayerNorm,
      InitializeF embedLayerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[decoderInputEmbedDim] (Double -> embedLayerNorm))),
      WithDeviceC device (WithDataTypeF dataType (WithDimsF '[decoderInputEmbedDim] (Double -> embedLayerNorm))),
      WithDataTypeC dataType (WithDimsF '[decoderInputEmbedDim] (Double -> embedLayerNorm)),
      WithDimsC '[decoderInputEmbedDim] (Double -> embedLayerNorm)
    )
  HasInitializeTDEmbedLayerNormF _ 'Pegasus _ _ _ = ()

type family
  HasInitializeTDLayerNormF
    (layerNorm :: Type)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Constraint
  where
  HasInitializeTDLayerNormF layerNorm 'T5 device dataType decoderInputEmbedDim =
    ( HasInitialize layerNorm,
      InitializeF layerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[decoderInputEmbedDim] (Double -> layerNorm))),
      WithDeviceC device (WithDataTypeF dataType (WithDimsF '[decoderInputEmbedDim] (Double -> layerNorm))),
      WithDataTypeC dataType (WithDimsF '[decoderInputEmbedDim] (Double -> layerNorm)),
      WithDimsC '[decoderInputEmbedDim] (Double -> layerNorm)
    )
  HasInitializeTDLayerNormF _ 'BERT _ _ _ = ()
  HasInitializeTDLayerNormF layerNorm 'Pegasus device dataType decoderInputEmbedDim =
    ( HasInitialize layerNorm,
      InitializeF layerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[decoderInputEmbedDim] (Double -> layerNorm))),
      WithDeviceC device (WithDataTypeF dataType (WithDimsF '[decoderInputEmbedDim] (Double -> layerNorm))),
      WithDataTypeC dataType (WithDimsF '[decoderInputEmbedDim] (Double -> layerNorm)),
      WithDimsC '[decoderInputEmbedDim] (Double -> layerNorm)
    )

type family
  HasInitializeTDPosEncF
    (posEnc :: Type)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Constraint
  where
  HasInitializeTDPosEncF posEnc 'T5 device dataType headDim _ posEncDim =
    ( HasInitialize posEnc,
      InitializeF posEnc ~ WithDeviceF device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF headDim (Generator device -> (posEnc, Generator device))))),
      WithDeviceC device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF headDim (Generator device -> (posEnc, Generator device))))),
      WithDataTypeC dataType (WithDimF posEncDim (WithDimF headDim (Generator device -> (posEnc, Generator device)))),
      WithDimC posEncDim (WithDimF headDim (Generator device -> (posEnc, Generator device))),
      WithDimC headDim (Generator device -> (posEnc, Generator device))
    )
  HasInitializeTDPosEncF posEnc 'BERT device dataType _ decoderInputEmbedDim posEncDim =
    ( HasInitialize posEnc,
      InitializeF posEnc ~ WithDeviceF device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF decoderInputEmbedDim (Generator device -> (posEnc, Generator device))))),
      WithDeviceC device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF decoderInputEmbedDim (Generator device -> (posEnc, Generator device))))),
      WithDataTypeC dataType (WithDimF posEncDim (WithDimF decoderInputEmbedDim (Generator device -> (posEnc, Generator device)))),
      WithDimC posEncDim (WithDimF decoderInputEmbedDim (Generator device -> (posEnc, Generator device))),
      WithDimC decoderInputEmbedDim (Generator device -> (posEnc, Generator device))
    )
  HasInitializeTDPosEncF posEnc 'Pegasus device dataType _ decoderInputEmbedDim posEncDim =
    ( HasInitialize posEnc,
      InitializeF posEnc ~ WithDeviceF device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF decoderInputEmbedDim (Generator device -> (posEnc, Generator device))))),
      WithDeviceC device (WithDataTypeF dataType (WithDimF posEncDim (WithDimF decoderInputEmbedDim (Generator device -> (posEnc, Generator device))))),
      WithDataTypeC dataType (WithDimF posEncDim (WithDimF decoderInputEmbedDim (Generator device -> (posEnc, Generator device)))),
      WithDimC posEncDim (WithDimF decoderInputEmbedDim (Generator device -> (posEnc, Generator device))),
      WithDimC decoderInputEmbedDim (Generator device -> (posEnc, Generator device))
    )

instance
  ( SingI style,
    HasInitializeTransformerDecoderC numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP,
    stack ~ TDStackF numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP,
    HasInitialize stack,
    InitializeF stack ~ WithDeviceF device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))))))))),
    HasInitializeTransformerDecoderStackC stack device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP,
    embedLayerNorm ~ TDEmbedLayerNormF style device dataType decoderInputEmbedDim,
    HasInitializeTDEmbedLayerNormF embedLayerNorm style device dataType decoderInputEmbedDim,
    layerNorm ~ TDLayerNormF style device dataType decoderInputEmbedDim,
    HasInitializeTDLayerNormF layerNorm style device dataType decoderInputEmbedDim,
    dropout ~ TDDropoutF style dropoutP,
    HasInitialize dropout,
    InitializeF dropout ~ (dropoutP -> dropout),
    posEnc ~ TDPosEncF style device dataType headDim decoderInputEmbedDim posEncDim,
    HasInitializeTDPosEncF posEnc style device dataType headDim decoderInputEmbedDim posEncDim
  ) =>
  HasInitialize (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
  where
  type
    InitializeF (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP) =
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
                            decoderInputEmbedDim
                            ( WithDimF
                                encoderOutputEmbedDim
                                ( WithDimF
                                    ffnDim
                                    ( WithDimF
                                        posEncDim
                                        (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device))
                                    )
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
                        withDim @decoderInputEmbedDim $
                          \decoderInputEmbedDim ->
                            withDim @encoderOutputEmbedDim $
                              \encoderOutputEmbedDim ->
                                withDim @ffnDim $
                                  \ffnDim ->
                                    withDim @posEncDim @(dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device)) $
                                      \posEncDim -> go deviceType dType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim
    where
      go deviceType dType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP eps = runState $ do
        stack <-
          state $
            withoutDim @ffnDim @(dropoutP -> Double -> Generator device -> (stack, Generator device))
              ( withoutDim @encoderOutputEmbedDim
                  ( withoutDim @decoderInputEmbedDim
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
                      decoderInputEmbedDim
                  )
                  encoderOutputEmbedDim
              )
              ffnDim
              dropoutP
              eps
        let embedLayerNorm = case sing @style of
              ST5 -> ()
              SBERT ->
                withoutShape @('Shape '[decoderInputEmbedDim])
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @embedLayerNorm
                          )
                          deviceType
                      )
                      dType
                  )
                  [decoderInputEmbedDim]
                  eps
              SPegasus -> ()
        let layerNorm = case sing @style of
              ST5 ->
                withoutShape @('Shape '[decoderInputEmbedDim])
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @layerNorm
                          )
                          deviceType
                      )
                      dType
                  )
                  [decoderInputEmbedDim]
                  eps
              SBERT -> ()
              SPegasus ->
                withoutShape @('Shape '[decoderInputEmbedDim])
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @layerNorm
                          )
                          deviceType
                      )
                      dType
                  )
                  [decoderInputEmbedDim]
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
              withoutDim @decoderInputEmbedDim @(Generator device -> (posEnc, Generator device))
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
                decoderInputEmbedDim
            SPegasus ->
              withoutDim @decoderInputEmbedDim @(Generator device -> (posEnc, Generator device))
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
                decoderInputEmbedDim
        pure . TransformerDecoder $ GTransformerDecoder stack embedLayerNorm layerNorm dropout posEnc

lookupDecoder ::
  forall numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim decoderInputEmbedDim,
    KnownDim encoderOutputEmbedDim,
    KnownDim ffnDim,
    KnownDim posEncDim,
    Scalar dropoutP,
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP m
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
lookupDecoder dropoutP eps prefix =
  let stack ST5 = lookupDecoderStack dropoutP eps (prefix <> "block.")
      embedLayerNorm ST5 = pure @m ()
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "final_layer_norm.weight")
          <*> pure eps
      dropout ST5 = pure (initialize @(Dropout dropoutP) dropoutP)
      posEnc ST5 = fmap @m Embedding $ lookupTensor (prefix <> "block.0.layer.0.SelfAttention.relative_attention_bias.weight")
   in TransformerDecoder
        <$> ( GTransformerDecoder
                <$> stack (sing @style)
                <*> embedLayerNorm (sing @style)
                <*> layerNorm (sing @style)
                <*> dropout (sing @style)
                <*> posEnc (sing @style)
            )

-- | 'HasForward' instance for @TransformerDecoder numLayers 'T5@.
--
-- @
-- ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ encoderOutput │  │ decoderRelPos │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └───────┬───────┘  └───────┬───────┘  └──────────┬───────────┘  └─────────┬──────────┘
--        │                  │                  │                     │                        │
--        │                  │                  ▼                     │                        │
--        │                  │              tdPosEnc                  │                        │
--        │                  │                  ▼                     │                        │
--        │                  │              transpose                 │                        │
--        │                  │                  ▼                     ▼                        ▼
--        │                  │              transpose             unsqueeze                unsqueeze
--        ▼                  │                  │                     │                        │
--    tdDropout              │                  └────────►add◄────────┘                        │
--        ▼                  │                             │                                   │
--     tdStack◄──────────────┘◄────────────────────────────┘◄──────────────────────────────────┘
--        ▼
--   tdLayerNorm
--        ▼
--    tdDropout
--        │
--        ▼
--    ┌────────┐
--    │ output │
--    └────────┘
-- @
instance
  ( HasForward
      (Dropout dropoutP)
      decoderInput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TransformerDecoderStack numLayers 'T5 device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          'WithGradient
          ('Layout 'Dense <+> decoderRelPosLayout <+> decoderAttentionMaskLayout)
          (device <+> decoderRelPosDevice <+> decoderAttentionMaskDevice)
          (Seq (decoderRelPosDataType <+> 'DataType 'Int64) dataType <+> decoderAttentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      ( EmbeddingF
                          ('Shape '[posEncDim, headDim])
                          decoderRelPosShape
                      )
                  )
              )
              ( UnsqueezeF
                  ('SelectDim ('ByIndex 1))
                  decoderAttentionMaskShape
              )
          ),
        Tensor
          crossAttentionMaskRequiresGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          ( UnsqueezeF
              ('SelectDim ('ByIndex 1))
              crossAttentionMaskShape
          )
      )
      dropoutGeneratorOutput
      stackOutput
      stackGeneratorOutput,
    HasForward
      ( LayerNorm
          'WithoutBias
          device
          dataType
          ('Shape '[decoderInputEmbedDim])
      )
      stackOutput
      stackGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (Dropout dropoutP)
      layerNormOutput
      layerNormGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoder numLayers 'T5 device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( decoderInput,
      encoderOutput,
      Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) =
    let decoderRelPosBias =
          ireturn decoderRelPos
            >>>= IxState . forward tdPosEnc
            >>>= ireturn . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        decoderAttentionBias =
          decoderRelPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask)
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxState $
          ireturn decoderInput
            >>>= IxState . forward tdDropout
            >>>= ( \decoderInput' ->
                     decoderAttentionBias
                       >>>= ( \decoderAttentionBias' ->
                                IxState $
                                  forward
                                    tdStack
                                    ( decoderInput',
                                      encoderOutput,
                                      decoderAttentionBias',
                                      crossAttentionBias
                                    )
                            )
                 )
            >>>= IxState . forward tdLayerNorm
            >>>= IxState . forward tdDropout

-- | 'HasForward' instance for @TransformerDecoder numLayers 'BART@.
--
-- @
-- ┌──────────────┐  ┌────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ decoderPos │  │ encoderOutput │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └──────┬─────┘  └───────┬───────┘  └──────────┬───────────┘  └──────────┬─────────┘
--        │                 │                │                     │                         │
--        │                 ▼                │                     │                         │
--        │             tdPosEnc             │                     │                         │
--        │                 │                │                     │                         │
--        └──────►add◄──────┘                │                     │                         │
--                 │                         │                     │                         │
--                 ▼                         │                     │                         │
--          tdEmbedLayerNorm                 │                     │                         │
--                 ▼                         │                     ▼                         ▼
--             tdDropout                     │                 unsqueeze                 unsqueeze
--                 ▼                         │                     │                         │
--              tdStack◄─────────────────────┘◄────────────────────┘◄────────────────────────┘
--                 │
--                 ▼
--            ┌────────┐
--            │ output │
--            └────────┘
-- @

-- | 'HasForward' instance for @TransformerDecoder numLayers 'MBART@.
--
-- @
-- ┌──────────────┐  ┌────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ decoderPos │  │ encoderOutput │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └──────┬─────┘  └───────┬───────┘  └──────────┬───────────┘  └──────────┬─────────┘
--        │                 │                │                     │                         │
--        │                 ▼                │                     │                         │
--        │             tdPosEnc             │                     │                         │
--        │                 │                │                     │                         │
--        └──────►add◄──────┘                │                     │                         │
--                 │                         │                     │                         │
--                 ▼                         │                     │                         │
--          tdEmbedLayerNorm                 │                     │                         │
--                 ▼                         │                     ▼                         ▼
--             tdDropout                     │                 unsqueeze                 unsqueeze
--                 ▼                         │                     │                         │
--              tdStack◄─────────────────────┘◄────────────────────┘◄────────────────────────┘
--                 ▼
--            tdLayerNorm
--                 │
--                 ▼
--            ┌────────┐
--            │ output │
--            └────────┘
-- @

-- | 'HasForward' instance for @TransformerDecoder numLayers 'Pegasus@.
--
-- @
-- ┌──────────────┐  ┌────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ decoderPos │  │ encoderOutput │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └──────┬─────┘  └───────┬───────┘  └──────────┬───────────┘  └──────────┬─────────┘
--        │                 │                │                     │                         │
--        │                 ▼                │                     │                         │
--        │             tdPosEnc             │                     │                         │
--        │                 │                │                     │                         │
--        └──────►add◄──────┘                │                     │                         │
--                 │                         │                     │                         │
--                 ▼                         │                     ▼                         ▼
--             tdDropout                     │                 unsqueeze                 unsqueeze
--                 ▼                         │                     │                         │
--              tdStack◄─────────────────────┘◄────────────────────┘◄────────────────────────┘
--                 ▼
--            tdLayerNorm
--                 │
--                 ▼
--            ┌────────┐
--            │ output │
--            └────────┘
-- @
instance
  ( HasForward
      (TDDropoutF 'Pegasus dropoutP)
      ( Tensor
          'WithGradient
          (decoderInputLayout <+> 'Layout 'Dense <+> decoderPosLayout)
          (decoderInputDevice <+> device <+> decoderPosDevice)
          (decoderInputDataType <+> Seq (decoderPosDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF decoderInputShape (EmbeddingF ('Shape '[posEncDim, decoderInputEmbedDim]) decoderPosShape))
      )
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TDStackF numLayers 'Pegasus device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          decoderAttentionMaskRequiresGradient
          decoderAttentionMaskLayout
          decoderAttentionMaskDevice
          decoderAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) decoderAttentionMaskShape),
        Tensor
          crossAttentionMaskRequiresGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) crossAttentionMaskShape)
      )
      dropoutGeneratorOutput
      stackOutput
      generatorOutput,
    HasForward
      (TDLayerNormF 'Pegasus device dataType decoderInputEmbedDim)
      stackOutput
      generatorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoder numLayers 'Pegasus device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      encoderOutput,
      Tensor decoderPosRequiresGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask) =
    let decoderAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxState $
          ireturn decoderPos
            >>>= IxState . forward tdPosEnc
            >>>= ireturn . (decoderInput `add`)
            >>>= IxState . forward tdDropout
            >>>= ( \decoderInput' ->
                     IxState $
                       forward
                         tdStack
                         ( decoderInput',
                           encoderOutput,
                           decoderAttentionBias,
                           crossAttentionBias
                         )
                 )
            >>>= IxState . forward tdLayerNorm