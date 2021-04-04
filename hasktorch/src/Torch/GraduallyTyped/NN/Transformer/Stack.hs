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
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Singletons (SingI)
import GHC.TypeLits (Nat, Symbol, type (+), type (-), type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.Block (HasInitializeTransformerBlockC, TransformerBlock, lookupBlock)
import Torch.GraduallyTyped.NN.Transformer.Type (TensorDict, TransformerStyle)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), KnownDim, Name (..), Size (..), WithDimC (..))

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
  (stack :: Type)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device)))),
    WithDimC queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (stack, Generator device))
  )

instance
  HasInitializeTransformerStackC (TransformerStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =>
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
  ( HasInitialize (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP),
    HasInitializeTransformerBlockC style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitializeTransformerStackC (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitializeTransformerStackC (TransformerStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
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
  HasLookupStack
    (n :: Nat)
    (isCons :: Bool)
    (numLayers :: Nat)
    style
    device
    dataType
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    ffnDim
    dropoutP
    m
  where
  lookupStack' ::
    dropoutP ->
    Double ->
    String ->
    Integer ->
    m (TransformerStack n style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)

instance
  Applicative m =>
  HasLookupStack 0 'False numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m
  where
  lookupStack' _ _ _ _ = pure TransformerStackNil

instance
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
    Scalar dropoutP,
    HasLookupStack
      (n - 1)
      (1 <=? (n - 1))
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
      m
  ) =>
  HasLookupStack n 'True numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m
  where
  lookupStack' dropoutP eps prefix n =
    TransformerStackCons
      <$> lookupBlock dropoutP eps (prefix <> show n <> ".")
      <*> lookupStack' @(n - 1) @(1 <=? (n - 1)) @numLayers @style @device @dataType @headDim @headEmbedDim @embedDim @queryEmbedDim @ffnDim @dropoutP dropoutP eps prefix (n + 1)

lookupStack ::
  forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
lookupStack dropoutP eps prefix = lookupStack' @numLayers @(1 <=? numLayers) @numLayers @style @device @dataType @headDim @headEmbedDim @embedDim @queryEmbedDim @ffnDim @dropoutP dropoutP eps prefix 0

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
