{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
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

module Torch.GraduallyTyped.NN.Transformer.DecoderStack where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol, type (+), type (-), type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (HasInitializeTransformerDecoderBlockC, TransformerDecoderBlock)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Size (..), WithDimC (..))

-- | Transformer decoder stack.
data
  TransformerDecoderStack
    (numLayers :: Nat)
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
  TransformerDecoderStackNil ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
    TransformerDecoderStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
  TransformerDecoderStackCons ::
    forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
    { -- | decoder layer block
      tdsBlock :: TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
      -- | remaining decoder stack
      tdsStack :: TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
    } ->
    TransformerDecoderStack (numLayers + 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

class
  HasInitializeTransformerDecoderStack
    (isCons :: Bool)
    (numLayers :: Nat)
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
  initializeTransformerDecoderStack ::
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
                                  (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
                              )
                          )
                      )
                  )
              )
          )
      )

type HasInitializeTransformerDecoderStackC stack device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device)))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device)))),
    WithDimC keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))
  )

instance
  HasInitializeTransformerDecoderStackC (TransformerDecoderStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =>
  HasInitializeTransformerDecoderStack 'False 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
  where
  initializeTransformerDecoderStack =
    withDevice @device $
      \_deviceType ->
        withDataType @dataType $
          \_dType ->
            withDim @headDim $
              \_headDim ->
                withDim @headEmbedDim $
                  \_headEmbedDim ->
                    withDim @embedDim $
                      \_embedDim ->
                        withDim @queryEmbedDim $
                          \_queryEmbedDim ->
                            withDim @keyEmbedDim $
                              \_keyEmbedDim ->
                                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoderStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)) $
                                  \_ffnDim _dropoutP _eps g -> (TransformerDecoderStackNil, g)

instance
  ( HasInitialize (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP),
    HasInitializeTransformerDecoderBlockC style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
    HasInitializeTransformerDecoderStackC (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
    HasInitializeTransformerDecoderStackC (TransformerDecoderStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
    HasInitialize (TransformerDecoderStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
  ) =>
  HasInitializeTransformerDecoderStack 'True numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
  where
  initializeTransformerDecoderStack =
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
                                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)) $
                                  \ffnDim -> go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps = runState $ do
        decoderStack <-
          state $
            withoutDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoderStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
              ( withoutDim @keyEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerDecoderStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
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
              )
              ffnDim
              dropoutP
              eps
        decoderBlock <-
          state $
            withoutDim @ffnDim
              ( withoutDim @keyEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
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
              )
              ffnDim
              dropoutP
              eps
        pure $ TransformerDecoderStackCons decoderBlock decoderStack

instance
  HasInitializeTransformerDecoderStack (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =>
  HasInitialize (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) =
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
                                    (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
                                )
                            )
                        )
                    )
                )
            )
        )
  initialize = initializeTransformerDecoderStack @(1 <=? numLayers) @numLayers @style @device @dataType @headDim @headEmbedDim @embedDim @queryEmbedDim @keyEmbedDim @ffnDim @dropoutP

class
  HasForwardTransformerDecoderStack
    (isCons :: Bool)
    (isNotFirstLayer :: Bool)
    (numLayers :: Nat)
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
    (query :: Type)
    (key :: Type)
    (decoderAttentionBias :: Type)
    (crossAttentionBias :: Type)
    (generator :: Type)
    (output :: Type)
    (generatorOutput :: Type)
    | isCons isNotFirstLayer numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP query key decoderAttentionBias crossAttentionBias generator -> output,
      isCons isNotFirstLayer numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP query key decoderAttentionBias crossAttentionBias generator -> generatorOutput
  where
  forwardTransformerDecoderStack ::
    Maybe
      ( TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP ->
        (query, key, decoderAttentionBias, crossAttentionBias) ->
        generator ->
        (query, generator)
      ) ->
    TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP ->
    (query, key, decoderAttentionBias, crossAttentionBias) ->
    generator ->
    (output, generatorOutput)

instance
  HasForwardTransformerDecoderStack
    'False
    isNotFirstLayer
    0
    style
    device
    dataType
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    keyEmbedDim
    ffnDim
    dropoutP
    query
    key
    decoderAttentionBias
    crossAttentionBias
    generator
    query
    generator
  where
  forwardTransformerDecoderStack _ TransformerDecoderStackNil (query, _key, _decoderAttentionBias, _crossAttentionBias) g = (query, g)

instance
  ( HasForward
      (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      (query, key, decoderAttentionBias, crossAttentionBias)
      generator
      blockOutput
      blockGeneratorOutput,
    HasForward
      (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      (blockOutput, key, decoderAttentionBias, crossAttentionBias)
      blockGeneratorOutput
      blockOutput
      blockGeneratorOutput,
    HasForwardTransformerDecoderStack
      (1 <=? numLayers - 1)
      'True
      (numLayers - 1)
      style
      device
      dataType
      headDim
      headEmbedDim
      embedDim
      queryEmbedDim
      keyEmbedDim
      ffnDim
      dropoutP
      blockOutput
      key
      decoderAttentionBias
      crossAttentionBias
      blockGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForwardTransformerDecoderStack
    'True
    'False
    numLayers
    style
    device
    dataType
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    keyEmbedDim
    ffnDim
    dropoutP
    query
    key
    decoderAttentionBias
    crossAttentionBias
    generator
    output
    generatorOutput
  where
  forwardTransformerDecoderStack _ (TransformerDecoderStackCons decoderBlock decoderStack) (query, key, decoderAttentionBias, crossAttentionBias) =
    runIxState $
      ireturn (query, key, decoderAttentionBias, crossAttentionBias)
        >>>= IxState . forward decoderBlock
        >>>= ( \query' ->
                 IxState $
                   forwardTransformerDecoderStack
                     @(1 <=? numLayers - 1)
                     @'True
                     @(numLayers - 1)
                     (Just forward)
                     decoderStack
                     (query', key, decoderAttentionBias, crossAttentionBias)
             )

instance
  HasForwardTransformerDecoderStack
    (1 <=? numLayers - 1)
    'True
    (numLayers - 1)
    style
    device
    dataType
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    keyEmbedDim
    ffnDim
    dropoutP
    query
    key
    decoderAttentionBias
    crossAttentionBias
    generator
    query
    generator =>
  HasForwardTransformerDecoderStack
    'True
    'True
    numLayers
    style
    device
    dataType
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    keyEmbedDim
    ffnDim
    dropoutP
    query
    key
    decoderAttentionBias
    crossAttentionBias
    generator
    query
    generator
  where
  forwardTransformerDecoderStack (Just f) (TransformerDecoderStackCons decoderBlock decoderStack) (query, key, decoderAttentionBias, crossAttentionBias) =
    runIxState $
      ireturn (query, key, decoderAttentionBias, crossAttentionBias)
        >>>= IxState . f decoderBlock
        >>>= ( \query' ->
                 IxState $
                   forwardTransformerDecoderStack
                     @(1 <=? numLayers - 1)
                     @'True
                     @(numLayers - 1)
                     (Just f)
                     decoderStack
                     (query', key, decoderAttentionBias, crossAttentionBias)
             )

-- | 'HasForward' instance for 'TransformerDecoderStack'.
--
-- @
-- ┌───────┐  ┌─────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ query │  │ key │  │ decoderAttentionBias │  │ crossAttentionBias │
-- └───┬───┘  └──┬──┘  └──────────┬───────────┘  └─────────┬──────────┘
--     │         │                │                        │
--     ▼         │                │                        │
--  tdsBlock◄────┤◄───────────────┤◄───────────────────────┤
--     ▼         │                │                        │
--  tdsBlock◄────┤◄───────────────┤◄───────────────────────┤
--     ▼         │                │                        │
--    ...       ...              ...                      ...
--     ▼         │                │                        │
--  tdsBlock◄────┘◄───────────────┘◄───────────────────────┘
--     │
--     ▼
-- ┌───────┐
-- │ query │
-- └───────┘
-- @
instance
  HasForwardTransformerDecoderStack
    (1 <=? numLayers)
    'False
    numLayers
    style
    device
    dataType
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    keyEmbedDim
    ffnDim
    dropoutP
    query
    key
    decoderAttentionBias
    crossAttentionBias
    generator
    output
    generatorOutput =>
  HasForward
    (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    output
    generatorOutput
  where
  forward = forwardTransformerDecoderStack @(1 <=? numLayers) @'False Nothing
