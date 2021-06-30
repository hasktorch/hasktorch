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
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.Stack where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Singletons (SingI)
import qualified Data.Vector.Sized as VS
import GHC.TypeLits (KnownNat, Nat, Symbol, type (+), type (-), type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.Block (TransformerBlock, lookupBlock)
import Torch.GraduallyTyped.NN.Transformer.Type (TensorDict, TransformerStyle)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), KnownDim, Name (..), SDim, Size (..))

-- -- | Transformer encoder stack.
-- data
--   TransformerStack
--     (numLayers :: Nat)
--     (style :: TransformerStyle)
--     (device :: Device (DeviceType Nat))
--     (dataType :: DataType DType)
--     (headDim :: Dim (Name Symbol) (Size Nat))
--     (headEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (embedDim :: Dim (Name Symbol) (Size Nat))
--     (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (ffnDim :: Dim (Name Symbol) (Size Nat))
--     (dropoutP :: Type)
--   where
--   TransformerStackNil ::
--     forall style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
--     TransformerStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
--   TransformerStackCons ::
--     forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
--     { -- | encoder layer block
--       tsBlock :: TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
--       -- | remaining encoder stack
--       tsStack :: TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
--     } ->
--     TransformerStack (numLayers + 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

-- | Transformer encoder stack.
newtype
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
  TransformerStack ::
    forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    VS.Vector numLayers (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) ->
    TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

instance
  ( KnownNat numLayers,
    HasInitialize
      (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      input
      generator
      generator',
    HasInitialize
      (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      input
      generator'
      generator'
  ) =>
  HasInitialize
    (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    input
    generator
    generator'
  where
  initialize input g =
    let (v, g') = initialize input g
     in (TransformerStack v, g')

instance
  ( HasForward
      (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      input
      generator
      output
      generatorOutput,
    HasForward
      (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      output
      generatorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    input
    generator
    output
    generatorOutput
  where
  forward (TransformerStack v) = forward v

-- class
--   HasInitializeTransformerStack
--     (isCons :: Bool)
--     (numLayers :: Nat)
--     (style :: TransformerStyle)
--     (device :: Device (DeviceType Nat))
--     (dataType :: DataType DType)
--     (headDim :: Dim (Name Symbol) (Size Nat))
--     (headEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (embedDim :: Dim (Name Symbol) (Size Nat))
--     (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (ffnDim :: Dim (Name Symbol) (Size Nat))
--     (dropoutP :: Type)
--   where
--   initializeTransformerStack ::
--     SDevice device ->
--     SDataType dataType ->
--     SDim headDim ->
--     SDim headEmbedDim ->
--     SDim embedDim ->
--     SDim queryEmbedDim ->
--     SDim ffnDim ->
--     dropoutP ->
--     Double ->
--     Generator device ->
--     (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)

-- instance HasInitializeTransformerStack 'False 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP where
--   initializeTransformerStack _ _ _ _ _ _ _ _ _ g = (TransformerStackNil, g)

-- instance
--   ( HasInitialize (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP),
--     HasInitialize (TransformerStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
--   ) =>
--   HasInitializeTransformerStack 'True numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
--   where
--   initializeTransformerStack device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps =
--     runState $ do
--       stack <-
--         state $
--           initialize @(TransformerStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
--             device
--             dataType
--             headDim
--             headEmbedDim
--             embedDim
--             queryEmbedDim
--             ffnDim
--             dropoutP
--             eps
--       block <-
--         state $
--           initialize @(TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
--             device
--             dataType
--             headDim
--             headEmbedDim
--             embedDim
--             queryEmbedDim
--             ffnDim
--             dropoutP
--             eps
--       pure $ TransformerStackCons block stack

-- instance
--   HasInitializeTransformerStack (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =>
--   HasInitialize (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
--   where
--   type
--     InitializeF (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) =
--       SDevice device ->
--       SDataType dataType ->
--       SDim headDim ->
--       SDim headEmbedDim ->
--       SDim embedDim ->
--       SDim queryEmbedDim ->
--       SDim ffnDim ->
--       dropoutP ->
--       Double ->
--       Generator device ->
--       (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)
--   initialize = initializeTransformerStack @(1 <=? numLayers) @numLayers @style @device @dataType @headDim @headEmbedDim @embedDim @queryEmbedDim @ffnDim @dropoutP

-- class
--   HasLookupStack
--     (n :: Nat)
--     (isCons :: Bool)
--     (numLayers :: Nat)
--     style
--     device
--     dataType
--     headDim
--     headEmbedDim
--     embedDim
--     queryEmbedDim
--     ffnDim
--     dropoutP
--     m
--   where
--   lookupStack' ::
--     SDim headDim ->
--     SDim headEmbedDim ->
--     SDim embedDim ->
--     dropoutP ->
--     Double ->
--     String ->
--     Integer ->
--     m (TransformerStack n style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)

-- instance
--   Applicative m =>
--   HasLookupStack 0 'False numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m
--   where
--   lookupStack' _ _ _ _ _ _ _ = pure TransformerStackNil

-- instance
--   ( SingI style,
--     MonadReader TensorDict m,
--     MonadIO m,
--     MonadFail m,
--     KnownDevice device,
--     KnownDataType dataType,
--     KnownDim headDim,
--     KnownDim headEmbedDim,
--     KnownDim embedDim,
--     KnownDim queryEmbedDim,
--     KnownDim ffnDim,
--     Scalar dropoutP,
--     HasLookupStack
--       (n - 1)
--       (1 <=? (n - 1))
--       numLayers
--       style
--       device
--       dataType
--       headDim
--       headEmbedDim
--       embedDim
--       queryEmbedDim
--       ffnDim
--       dropoutP
--       m
--   ) =>
--   HasLookupStack n 'True numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m
--   where
--   lookupStack' headDim headEmbedDim embedDim dropoutP eps prefix n =
--     TransformerStackCons
--       <$> lookupBlock headDim headEmbedDim embedDim dropoutP eps (prefix <> show n <> ".")
--       <*> lookupStack'
--         @(n - 1)
--         @(1 <=? (n - 1))
--         @numLayers
--         @style
--         @device
--         @dataType
--         @headDim
--         @headEmbedDim
--         @embedDim
--         @queryEmbedDim
--         @ffnDim
--         @dropoutP
--         headDim
--         headEmbedDim
--         embedDim
--         dropoutP
--         eps
--         prefix
--         (n + 1)

-- lookupStack ::
--   forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m.
--   ( SingI style,
--     MonadReader TensorDict m,
--     MonadIO m,
--     MonadFail m,
--     HasLookupStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m
--   ) =>
--   SDim headDim ->
--   SDim headEmbedDim ->
--   SDim embedDim ->
--   dropoutP ->
--   Double ->
--   String ->
--   m (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
-- lookupStack headDim headEmbedDim embedDim dropoutP eps prefix =
--   lookupStack'
--     @numLayers
--     @(1 <=? numLayers)
--     @numLayers
--     @style
--     @device
--     @dataType
--     @headDim
--     @headEmbedDim
--     @embedDim
--     @queryEmbedDim
--     @ffnDim
--     @dropoutP
--     headDim
--     headEmbedDim
--     embedDim
--     dropoutP
--     eps
--     prefix
--     0

-- class
--   HasForwardTransformerStack
--     (isCons :: Bool)
--     (isNotFirstLayer :: Bool)
--     (numLayers :: Nat)
--     (style :: TransformerStyle)
--     (device :: Device (DeviceType Nat))
--     (dataType :: DataType DType)
--     (headDim :: Dim (Name Symbol) (Size Nat))
--     (headEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (embedDim :: Dim (Name Symbol) (Size Nat))
--     (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (ffnDim :: Dim (Name Symbol) (Size Nat))
--     (dropoutP :: Type)
--     (query :: Type)
--     (attentionBias :: Type)
--     (generator :: Type)
--     (output :: Type)
--     (generatorOutput :: Type)
--     | isCons isNotFirstLayer numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP query attentionBias generator -> output,
--       isCons isNotFirstLayer numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP query attentionBias generator -> generatorOutput
--   where
--   forwardTransformerStack ::
--     Maybe
--       ( TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP ->
--         (query, attentionBias) ->
--         generator ->
--         (query, generator)
--       ) ->
--     TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP ->
--     (query, attentionBias) ->
--     generator ->
--     (output, generatorOutput)

-- instance
--   HasForwardTransformerStack
--     'False
--     isNotFirstLayer
--     0
--     style
--     device
--     dataType
--     headDim
--     headEmbedDim
--     embedDim
--     queryEmbedDim
--     ffnDim
--     dropoutP
--     query
--     attentionBias
--     generator
--     query
--     generator
--   where
--   forwardTransformerStack _ TransformerStackNil (query, _attentionBias) g = (query, g)

-- instance
--   ( HasForward
--       (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
--       (query, attentionBias)
--       generator
--       blockOutput
--       blockGeneratorOutput,
--     HasForward
--       (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
--       (blockOutput, attentionBias)
--       blockGeneratorOutput
--       blockOutput
--       blockGeneratorOutput,
--     HasForwardTransformerStack
--       (1 <=? numLayers - 1)
--       'True
--       (numLayers - 1)
--       style
--       device
--       dataType
--       headDim
--       headEmbedDim
--       embedDim
--       queryEmbedDim
--       ffnDim
--       dropoutP
--       blockOutput
--       attentionBias
--       blockGeneratorOutput
--       output
--       generatorOutput
--   ) =>
--   HasForwardTransformerStack
--     'True
--     'False
--     numLayers
--     style
--     device
--     dataType
--     headDim
--     headEmbedDim
--     embedDim
--     queryEmbedDim
--     ffnDim
--     dropoutP
--     query
--     attentionBias
--     generator
--     output
--     generatorOutput
--   where
--   forwardTransformerStack _ (TransformerStackCons block stack) (query, attentionBias) =
--     runIxState $
--       ireturn (query, attentionBias)
--         >>>= IxState . forward block
--         >>>= ( \query' ->
--                  IxState $
--                    forwardTransformerStack
--                      @(1 <=? numLayers - 1)
--                      @'True
--                      @(numLayers - 1)
--                      (Just forward)
--                      stack
--                      (query', attentionBias)
--              )

-- instance
--   HasForwardTransformerStack
--     (1 <=? numLayers - 1)
--     'True
--     (numLayers - 1)
--     style
--     device
--     dataType
--     headDim
--     headEmbedDim
--     embedDim
--     queryEmbedDim
--     ffnDim
--     dropoutP
--     query
--     attentionBias
--     generator
--     query
--     generator =>
--   HasForwardTransformerStack
--     'True
--     'True
--     numLayers
--     style
--     device
--     dataType
--     headDim
--     headEmbedDim
--     embedDim
--     queryEmbedDim
--     ffnDim
--     dropoutP
--     query
--     attentionBias
--     generator
--     query
--     generator
--   where
--   forwardTransformerStack (Just f) (TransformerStackCons block stack) (query, attentionBias) =
--     runIxState $
--       ireturn (query, attentionBias)
--         >>>= IxState . f block
--         >>>= ( \query' ->
--                  IxState $
--                    forwardTransformerStack
--                      @(1 <=? numLayers - 1)
--                      @'True
--                      @(numLayers - 1)
--                      (Just f)
--                      stack
--                      (query', attentionBias)
--              )

-- -- | 'HasForward' instance for 'TransformerStack'.
-- --
-- -- @
-- -- ┌───────┐  ┌───────────────┐
-- -- │ query │  │ attentionBias │
-- -- └───┬───┘  └───────┬───────┘
-- --     │              │
-- --     ▼              │
-- --  tsBlock◄──────────┤
-- --     ▼              │
-- --  tsBlock◄──────────┤
-- --     ▼              │
-- --    ...            ...
-- --     ▼              │
-- --  tsBlock◄──────────┘
-- --     │
-- --     ▼
-- -- ┌───────┐
-- -- │ query │
-- -- └───────┘
-- -- @
-- instance
--   HasForwardTransformerStack
--     (1 <=? numLayers)
--     'False
--     numLayers
--     style
--     device
--     dataType
--     headDim
--     headEmbedDim
--     embedDim
--     queryEmbedDim
--     ffnDim
--     dropoutP
--     query
--     attentionBias
--     generator
--     output
--     generatorOutput =>
--   HasForward
--     (TransformerStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
--     (query, attentionBias)
--     generator
--     output
--     generatorOutput
--   where
--   forward = forwardTransformerStack @(1 <=? numLayers) @'False Nothing
