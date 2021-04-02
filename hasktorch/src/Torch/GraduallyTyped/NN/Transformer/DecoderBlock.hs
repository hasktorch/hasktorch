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

module Torch.GraduallyTyped.NN.Transformer.DecoderBlock where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention, HasInitializeCrossAttentionC)
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (HasInitializeTransformerFeedForwardNetworkC, TransformerFeedForwardNetwork)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (HasInitializeSelfAttentionC, SelfAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

-- | Transformer decoder block consisting of self-attention,
-- cross-attention, and a feed-forward network.
data
  TransformerDecoderBlock
    (style :: TransformerStyle)
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
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
    { -- | self-attention layer
      tdbSelfAttention :: SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
      -- | cross-attention layer
      tdbCrossAttention :: CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
      -- | feed-forward network
      tdbFeedForwardNetwork :: TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

type HasInitializeTransformerDecoderBlockC style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  ( HasInitializeTransformerDecoderBlockC style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
    HasInitializeSelfAttentionC style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    HasInitialize (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP),
    InitializeF (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP) ~ WithDeviceF device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))))))),
    HasInitializeCrossAttentionC style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
    HasInitialize (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP),
    InitializeF (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP) ~ WithDeviceF device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)))))))),
    HasInitializeTransformerFeedForwardNetworkC style device dataType queryEmbedDim ffnDim dropoutP,
    HasInitialize (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP),
    InitializeF (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP) ~ WithDeviceF device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP, Generator device)))))
  ) =>
  HasInitialize (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) =
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
                                    (dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
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
                                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)) $
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
                                  ( initialize @(SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
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
                                      ( initialize @(CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
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
                          ( initialize @(TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
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

-- | 'HasForward' instance for 'TransformerDecoderBlock'.
--
-- @
-- ┌──────────────────────┐  ┌───────┐  ┌─────┐  ┌────────────────────┐
-- │ decoderAttentionBias │  │ query │  │ key │  │ crossAttentionBias │
-- └──────────┬───────────┘  └───┬───┘  └──┬──┘  └─────────┬──────────┘
--            │                  │         │               │
--            │                  ▼         │               │
--            └──────────►tdbSelfAttention │               │
--                               │         │               │
--                               ▼         ▼               │
--                            tdbCrossAttention◄───────────┘
--                               │
--                               ▼
--                     tdbFeedForwardNetwork
--                               │
--                               ▼
--                           ┌───────┐
--                           │ query │
--                           └───────┘
-- @
instance
  ( HasForward
      (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor decoderAttentionBiasRequiresGradient decoderAttentionBiasLayout decoderAttentionBiasDevice decoderAttentionBiasDataType decoderAttentionBiasShape
      )
      (Generator generatorDevice)
      selfAttentionOutput
      selfAttentionGeneratorOutput,
    HasForward
      (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      ( selfAttentionOutput,
        Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor crossAttentionBiasRequiresGradient crossAttentionBiasLayout crossAttentionBiasDevice crossAttentionBiasDataType crossAttentionBiasShape
      )
      selfAttentionGeneratorOutput
      crossAttentionOutput
      crossAttentionGeneratorOutput,
    HasForward
      (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
      crossAttentionOutput
      crossAttentionGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
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
