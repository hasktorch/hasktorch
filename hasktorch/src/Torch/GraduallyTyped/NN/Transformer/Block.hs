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

module Torch.GraduallyTyped.NN.Transformer.Block where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Singletons (SingI)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (HasInitializeTransformerFeedForwardNetworkC, TransformerFeedForwardNetwork, lookupTransformerFeedForwardNetwork)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (HasInitializeSelfAttentionC, SelfAttention, lookupSelfAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (TensorDict, TransformerStyle)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

-- | Transformer encoder block consisting of self-attention and a feed-forward network.
--
-- TODO: Some transformers use LayerDrop, see https://arxiv.org/abs/1909.11556, during training.
-- To support this, we will need a layer wrapper that is either the identity function or the wrapped layer
-- based on a uniformly random draw from a supplied generator.
-- Complications will arise due to the gradual typing...
data
  TransformerBlock
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerBlock ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    { -- | self-attention layer
      tbSelfAttention :: SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
      -- | feed-forward network
      tbFeedForwardNetwork :: TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

type HasInitializeTransformerBlockC style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  ( HasInitializeTransformerBlockC style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitialize (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP),
    InitializeF (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP) ~ WithDeviceF device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))))))),
    HasInitializeSelfAttentionC style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    HasInitializeTransformerFeedForwardNetworkC style device dataType queryEmbedDim ffnDim dropoutP,
    HasInitialize (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP),
    InitializeF (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP) ~ WithDeviceF device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP, Generator device)))))
  ) =>
  HasInitialize (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) =
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
                                ffnDim
                                (dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
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
                            withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)) $
                              \ffnDim ->
                                go deviceType dType headDim headEmbedDim embedDim queryEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps = runState $ do
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
        pure $ TransformerBlock selfAttention feedForwardNetwork

lookupBlock ::
  forall style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim queryEmbedDim,
    KnownDim ffnDim,
    Scalar dropoutP
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
lookupBlock dropoutP eps prefix =
  TransformerBlock
    <$> lookupSelfAttention dropoutP eps (prefix <> "layer.0.")
    <*> lookupTransformerFeedForwardNetwork dropoutP eps (prefix <> "layer.1.")

-- | 'HasForward' instance for 'TransformerBlock'.
--
-- @
--      ┌───────┐  ┌───────────────┐
--      │ query │  │ attentionBias │
--      └───┬───┘  └───────┬───────┘
--          │              │
--          ▼              │
--   tbSelfAttention◄──────┘
--          ▼
-- tbFeedForwardNetwork
--          │
--          ▼
--      ┌───────┐
--      │ query │
--      └───────┘
-- @
instance
  ( HasForward
      (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
      )
      (Generator generatorDevice)
      selfAttentionOutput
      selfAttentionGeneratorOutput,
    HasForward
      (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
      selfAttentionOutput
      selfAttentionGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward TransformerBlock {..} (query, attentionBias) =
    runIxState $
      ireturn (query, attentionBias)
        >>>= IxState . forward tbSelfAttention
        >>>= IxState . forward tbFeedForwardNetwork
