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

module Torch.GraduallyTyped.NN.Transformer.Stack where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol, type (+), type (-), type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.Block (HasInitializeTransformerBlockC, TransformerBlock)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Size (..), WithDimC (..))

-- | Transformer encoder stack.
data
  TransformerStack
    (numLayers :: Nat)
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
  TransformerStackNil ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    TransformerStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
  TransformerStackCons ::
    forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    { -- | encoder layer block
      tsBlock :: TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
      -- | remaining encoder stack
      tsStack :: TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
    } ->
    TransformerStack (numLayers + 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

class
  HasInitializeTransformerStack
    (isCons :: Bool)
    (numLayers :: Nat)
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
  initializeTransformerStack ::
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
                              (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
                          )
                      )
                  )
              )
          )
      )

type HasInitializeTransformerStackC
  (numLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerStackC 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =>
  HasInitializeTransformerStack 'False 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
  where
  initializeTransformerStack =
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
                            withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)) $
                              \_ffnDim _dropoutP _eps g -> (TransformerStackNil, g)

instance
  ( HasInitializeTransformerBlockC style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitializeTransformerStackC numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitializeTransformerStackC (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitialize (TransformerStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
  ) =>
  HasInitializeTransformerStack 'True numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
  where
  initializeTransformerStack =
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
                            withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)) $
                              \ffnDim -> go deviceType dType headDim headEmbedDim embedDim queryEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps = runState $ do
        stack <-
          state $
            withoutDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
              ( withoutDim @queryEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
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
              ffnDim
              dropoutP
              eps
        block <-
          state $
            withoutDim @ffnDim
              ( withoutDim @queryEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
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
              ffnDim
              dropoutP
              eps
        pure $ TransformerStackCons block stack

instance
  HasInitializeTransformerStack (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =>
  HasInitialize (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) =
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
                                (dropoutP -> Double -> Generator device -> (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
                            )
                        )
                    )
                )
            )
        )
  initialize = initializeTransformerStack @(1 <=? numLayers) @numLayers @style @device @dataType @headDim @headEmbedDim @embedDim @queryEmbedDim @ffnDim @dropoutP

class
  HasForwardTransformerStack
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
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
    (query :: Type)
    (attentionBias :: Type)
    (generator :: Type)
    (output :: Type)
    (generatorOutput :: Type)
    | isCons isNotFirstLayer numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP query attentionBias generator -> output,
      isCons isNotFirstLayer numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP query attentionBias generator -> generatorOutput
  where
  forwardTransformerStack ::
    Maybe
      ( TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP ->
        (query, attentionBias) ->
        generator ->
        (query, generator)
      ) ->
    TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP ->
    (query, attentionBias) ->
    generator ->
    (output, generatorOutput)

instance
  HasForwardTransformerStack
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
    ffnDim
    dropoutP
    query
    attentionBias
    generator
    query
    generator
  where
  forwardTransformerStack _ TransformerStackNil (query, _attentionBias) g = (query, g)

instance
  ( HasForward
      (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      (query, attentionBias)
      generator
      blockOutput
      blockGeneratorOutput,
    HasForward
      (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      (blockOutput, attentionBias)
      blockGeneratorOutput
      blockOutput
      blockGeneratorOutput,
    HasForwardTransformerStack
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
      ffnDim
      dropoutP
      blockOutput
      attentionBias
      blockGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForwardTransformerStack
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
    ffnDim
    dropoutP
    query
    attentionBias
    generator
    output
    generatorOutput
  where
  forwardTransformerStack _ (TransformerStackCons block stack) (query, attentionBias) =
    runIxState $
      ireturn (query, attentionBias)
        >>>= IxState . forward block
        >>>= ( \query' ->
                 IxState $
                   forwardTransformerStack
                     @(1 <=? numLayers - 1)
                     @'True
                     @(numLayers - 1)
                     (Just forward)
                     stack
                     (query', attentionBias)
             )

instance
  HasForwardTransformerStack
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
    ffnDim
    dropoutP
    query
    attentionBias
    generator
    query
    generator =>
  HasForwardTransformerStack
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
    ffnDim
    dropoutP
    query
    attentionBias
    generator
    query
    generator
  where
  forwardTransformerStack (Just f) (TransformerStackCons block stack) (query, attentionBias) =
    runIxState $
      ireturn (query, attentionBias)
        >>>= IxState . f block
        >>>= ( \query' ->
                 IxState $
                   forwardTransformerStack
                     @(1 <=? numLayers - 1)
                     @'True
                     @(numLayers - 1)
                     (Just f)
                     stack
                     (query', attentionBias)
             )

-- | 'HasForward' instance for 'TransformerStack'.
--
-- @
-- ┌───────┐  ┌───────────────┐
-- │ query │  │ attentionBias │
-- └───┬───┘  └───────┬───────┘
--     │              │
--     ▼              │
--  tsBlock◄──────────┤
--     ▼              │
--  tsBlock◄──────────┤
--     ▼              │
--    ...            ...
--     ▼              │
--  tsBlock◄──────────┘
--     │
--     ▼
-- ┌───────┐
-- │ query │
-- └───────┘
-- @
instance
  HasForwardTransformerStack
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
    ffnDim
    dropoutP
    query
    attentionBias
    generator
    output
    generatorOutput =>
  HasForward
    (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    (query, attentionBias)
    generator
    output
    generatorOutput
  where
  forward = forwardTransformerStack @(1 <=? numLayers) @'False Nothing
