{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
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

module Torch.GraduallyTyped.NN.Transformer.DecoderBlock where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention, CrossAttentionOutputShape, HasInitializeCrossAttentionC)
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (FeedForwardNetworkOutputShape, HasInitializeTransformerFeedForwardNetworkC, TransformerFeedForwardNetwork)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (HasInitializeSelfAttentionC, SelfAttention, SelfAttentionOutputShape)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  TransformerDecoderBlock
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerDecoderBlock ::
    forall device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
    { tdbSelfAttention :: SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
      tdbCrossAttention :: CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
      tdbFeedForwardNetwork :: TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

type HasInitializeTransformerDecoderBlockC device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =
  ( HasInitializeSelfAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    HasInitializeCrossAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
    HasInitializeTransformerFeedForwardNetworkC device dataType queryEmbedDim ffnDim dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerDecoderBlockC device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =>
  HasInitialize (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) =
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
                            queryEmbedDim
                            ( WithDimF
                                keyEmbedDim
                                ( WithDimF
                                    ffnDim
                                    (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
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
                        withDim @queryEmbedDim $
                          \queryEmbedDim ->
                            withDim @keyEmbedDim $
                              \keyEmbedDim ->
                                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)) $
                                  \ffnDim ->
                                    go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps = runState $ do
        selfAttention <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @embedDim
                  ( withoutDim @headEmbedDim
                      ( withoutDim @headDim
                          ( withoutDataType @dataType
                              ( withoutDevice @device
                                  ( initialize @(SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
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
              queryEmbedDim
              dropoutP
              eps
        crossAttention <-
          state $
            withoutDim @keyEmbedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
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
                  queryEmbedDim
              )
              keyEmbedDim
              dropoutP
              eps
        feedForwardNetwork <-
          state $
            withoutDim @ffnDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP)
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              ffnDim
              dropoutP
              eps
        pure $ TransformerDecoderBlock selfAttention crossAttention feedForwardNetwork

instance
  ( HasForward
      (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor decoderAttentionBiasRequiresGradient decoderAttentionBiasLayout decoderAttentionBiasDevice decoderAttentionBiasDataType decoderAttentionBiasShape
      )
      (Generator generatorDevice)
      selfAttentionOutput
      selfAttentionGeneratorOutput,
    HasForward
      (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      ( selfAttentionOutput,
        Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor crossAttentionBiasRequiresGradient crossAttentionBiasLayout crossAttentionBiasDevice crossAttentionBiasDataType crossAttentionBiasShape
      )
      selfAttentionGeneratorOutput
      crossAttentionOutput
      crossAttentionGeneratorOutput,
    HasForward
      (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP)
      crossAttentionOutput
      crossAttentionGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor decoderAttentionBiasRequiresGradient decoderAttentionBiasLayout decoderAttentionBiasDevice decoderAttentionBiasDataType decoderAttentionBiasShape,
      Tensor crossAttentionBiasRequiresGradient crossAttentionBiasLayout crossAttentionBiasDevice crossAttentionBiasDataType crossAttentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward TransformerDecoderBlock {..} (query, key, decoderAttentionBias, crossAttentionBias) =
    runIxState $
      ireturn (query, decoderAttentionBias)
        >>>= IxState . forward tdbSelfAttention
        >>>= (\query' -> IxState . forward tdbCrossAttention $ (query', key, crossAttentionBias))
        >>>= IxState . forward tdbFeedForwardNetwork
