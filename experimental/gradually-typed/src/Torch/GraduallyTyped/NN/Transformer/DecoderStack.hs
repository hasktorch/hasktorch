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
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -v2 -Wall -fconstraint-solver-iterations=2 #-}

module Torch.GraduallyTyped.NN.Transformer.DecoderStack where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Singletons (SingI)
import Data.Singletons.Prelude.List (SList (..))
import qualified Data.Vector as V
import qualified Data.Vector.Generic as VG
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import GHC.TypeLits (KnownNat, Nat, Symbol, type (+), type (-), type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (TransformerDecoderBlock, lookupDecoderBlock)
import Torch.GraduallyTyped.NN.Transformer.Type (TensorDict, TransformerStyle (..))
import Torch.GraduallyTyped.Random (Generator, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), SDim, SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)

-- | Transformer decoder stack.
-- data
--   TransformerDecoderStack
--     (numLayers :: Nat)
--     (style :: TransformerStyle)
--     (device :: Device (DeviceType Nat))
--     (dataType :: DataType DType)
--     (headDim :: Dim (Name Symbol) (Size Nat))
--     (headEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (embedDim :: Dim (Name Symbol) (Size Nat))
--     (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (ffnDim :: Dim (Name Symbol) (Size Nat))
--     (dropoutP :: Type)
--   where
--   TransformerDecoderStackNil ::
--     forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
--     TransformerDecoderStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
--   TransformerDecoderStackCons ::
--     forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
--     { -- | decoder layer block
--       tdsBlock :: TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
--       -- | remaining decoder stack
--       tdsStack :: TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
--     } ->
--     TransformerDecoderStack (numLayers + 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

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
  TransformerDecoderStack ::
    forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
    VS.Vector numLayers (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) ->
    TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

instance
  ( KnownNat numLayers,
    HasInitialize
      (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      input
      generator
      generator',
    HasInitialize
      (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      input
      generator'
      generator'
  ) =>
  HasInitialize
    (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    input
    generator
    generator'
  where
  initialize input g =
    let (v, g') = initialize input g
     in (TransformerDecoderStack v, g')

instance
  HasForward
    (TransformerDecoderStack 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    query
    generator
  where
  forward _ (query, _, _, _) g = (query, g)

instance
  HasForward
    (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    output
    generatorOutput =>
  HasForward
    (TransformerDecoderStack 1 style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoderStack (VGS.Vector v)) input g =
    let Just (block, _) = V.uncons v
     in forward block input g

instance
  {-# OVERLAPPABLE #-}
  ( HasForward
      (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      (query, key, decoderAttentionBias, crossAttentionBias)
      generator
      output
      generatorOutput,
    HasForward
      (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      (output, key, decoderAttentionBias, crossAttentionBias)
      generatorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoderStack n style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoderStack (VGS.Vector v)) (query, key, decoderAttentionBias, crossAttentionBias) g =
    let Just (block, blocks) = V.uncons v
     in V.foldl
          ( \(output, g') block' ->
              forward block' (output, key, decoderAttentionBias, crossAttentionBias) g'
          )
          (forward block (query, key, decoderAttentionBias, crossAttentionBias) g)
          blocks

-- class
--   HasInitializeTransformerDecoderStack
--     (isCons :: Bool)
--     (numLayers :: Nat)
--     (style :: TransformerStyle)
--     (device :: Device (DeviceType Nat))
--     (dataType :: DataType DType)
--     (headDim :: Dim (Name Symbol) (Size Nat))
--     (headEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (embedDim :: Dim (Name Symbol) (Size Nat))
--     (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (ffnDim :: Dim (Name Symbol) (Size Nat))
--     (dropoutP :: Type)
--   where
--   initializeTransformerDecoderStack ::
--     SDevice device ->
--     SDataType dataType ->
--     SDim headDim ->
--     SDim headEmbedDim ->
--     SDim embedDim ->
--     SDim queryEmbedDim ->
--     SDim keyEmbedDim ->
--     SDim ffnDim ->
--     dropoutP ->
--     Double ->
--     Generator device ->
--     (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)

-- instance HasInitializeTransformerDecoderStack 'False 0 style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP where
--   initializeTransformerDecoderStack _ _ _ _ _ _ _ _ _ _ g = (TransformerDecoderStackNil, g)

-- instance
--   ( HasInitialize (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP),
--     HasInitialize (TransformerDecoderStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
--   ) =>
--   HasInitializeTransformerDecoderStack 'True numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
--   where
--   initializeTransformerDecoderStack device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps =
--     runState $ do
--       decoderStack <-
--         state $
--           initialize
--             @(TransformerDecoderStack (numLayers - 1) style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
--             device
--             dataType
--             headDim
--             headEmbedDim
--             embedDim
--             queryEmbedDim
--             keyEmbedDim
--             ffnDim
--             dropoutP
--             eps
--       decoderBlock <-
--         state $
--           initialize @(TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
--             device
--             dataType
--             headDim
--             headEmbedDim
--             embedDim
--             queryEmbedDim
--             keyEmbedDim
--             ffnDim
--             dropoutP
--             eps
--       pure $ TransformerDecoderStackCons decoderBlock decoderStack

-- instance
--   HasInitializeTransformerDecoderStack (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =>
--   HasInitialize (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
--   where
--   type
--     InitializeF (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) =
--       SDevice device ->
--       SDataType dataType ->
--       SDim headDim ->
--       SDim headEmbedDim ->
--       SDim embedDim ->
--       SDim queryEmbedDim ->
--       SDim keyEmbedDim ->
--       SDim ffnDim ->
--       dropoutP ->
--       Double ->
--       Generator device ->
--       (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)
--   initialize = initializeTransformerDecoderStack @(1 <=? numLayers) @numLayers @style @device @dataType @headDim @headEmbedDim @embedDim @queryEmbedDim @keyEmbedDim @ffnDim @dropoutP

-- class
--   HasLookupDecoderStack
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
--     keyEmbedDim
--     ffnDim
--     dropoutP
--     m
--   where
--   lookupDecoderStack' ::
--     SDim headDim ->
--     SDim headEmbedDim ->
--     SDim embedDim ->
--     dropoutP ->
--     Double ->
--     String ->
--     Integer ->
--     m (TransformerDecoderStack n style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)

-- instance
--   Applicative m =>
--   HasLookupDecoderStack 0 'False numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP m
--   where
--   lookupDecoderStack' _ _ _ _ _ _ _ = pure TransformerDecoderStackNil

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
--     KnownDim keyEmbedDim,
--     KnownDim ffnDim,
--     Scalar dropoutP,
--     HasLookupDecoderStack
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
--       keyEmbedDim
--       ffnDim
--       dropoutP
--       m
--   ) =>
--   HasLookupDecoderStack n 'True numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP m
--   where
--   lookupDecoderStack' headDim headEmbedDim embedDim dropoutP eps prefix n =
--     TransformerDecoderStackCons
--       <$> lookupDecoderBlock headDim headEmbedDim embedDim dropoutP eps (prefix <> show n <> ".")
--       <*> lookupDecoderStack'
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
--         @keyEmbedDim
--         @ffnDim
--         @dropoutP
--         headDim
--         headEmbedDim
--         embedDim
--         dropoutP
--         eps
--         prefix
--         (n + 1)

-- lookupDecoderStack ::
--   forall numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP m.
--   ( SingI style,
--     MonadReader TensorDict m,
--     MonadIO m,
--     MonadFail m,
--     HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP m
--   ) =>
--   SDim headDim ->
--   SDim headEmbedDim ->
--   SDim embedDim ->
--   dropoutP ->
--   Double ->
--   String ->
--   m (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
-- lookupDecoderStack headDim headEmbedDim embedDim dropoutP eps prefix =
--   lookupDecoderStack'
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
--     @keyEmbedDim
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
--   HasForwardTransformerDecoderStack
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
--     (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
--     (ffnDim :: Dim (Name Symbol) (Size Nat))
--     (dropoutP :: Type)
--     (query :: Type)
--     (key :: Type)
--     (decoderAttentionBias :: Type)
--     (crossAttentionBias :: Type)
--     (generator :: Type)
--     (output :: Type)
--     (generatorOutput :: Type)
--     | isCons isNotFirstLayer numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP query key decoderAttentionBias crossAttentionBias generator -> output,
--       isCons isNotFirstLayer numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP query key decoderAttentionBias crossAttentionBias generator -> generatorOutput
--   where
--   forwardTransformerDecoderStack ::
--     Maybe
--       ( TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP ->
--         (query, key, decoderAttentionBias, crossAttentionBias) ->
--         generator ->
--         (query, generator)
--       ) ->
--     TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP ->
--     (query, key, decoderAttentionBias, crossAttentionBias) ->
--     generator ->
--     (output, generatorOutput)

-- instance
--   HasForwardTransformerDecoderStack
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
--     keyEmbedDim
--     ffnDim
--     dropoutP
--     query
--     key
--     decoderAttentionBias
--     crossAttentionBias
--     generator
--     query
--     generator
--   where
--   forwardTransformerDecoderStack _ TransformerDecoderStackNil (query, _key, _decoderAttentionBias, _crossAttentionBias) g = (query, g)

-- instance
--   ( HasForward
--       (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
--       (query, key, decoderAttentionBias, crossAttentionBias)
--       generator
--       blockOutput
--       blockGeneratorOutput,
--     HasForward
--       (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
--       (blockOutput, key, decoderAttentionBias, crossAttentionBias)
--       blockGeneratorOutput
--       blockOutput
--       blockGeneratorOutput,
--     HasForwardTransformerDecoderStack
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
--       keyEmbedDim
--       ffnDim
--       dropoutP
--       blockOutput
--       key
--       decoderAttentionBias
--       crossAttentionBias
--       blockGeneratorOutput
--       output
--       generatorOutput
--   ) =>
--   HasForwardTransformerDecoderStack
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
--     keyEmbedDim
--     ffnDim
--     dropoutP
--     query
--     key
--     decoderAttentionBias
--     crossAttentionBias
--     generator
--     output
--     generatorOutput
--   where
--   forwardTransformerDecoderStack _ (TransformerDecoderStackCons decoderBlock decoderStack) (query, key, decoderAttentionBias, crossAttentionBias) =
--     runIxState $
--       ireturn (query, key, decoderAttentionBias, crossAttentionBias)
--         >>>= IxState . forward decoderBlock
--         >>>= ( \query' ->
--                  IxState $
--                    forwardTransformerDecoderStack
--                      @(1 <=? numLayers - 1)
--                      @'True
--                      @(numLayers - 1)
--                      (Just forward)
--                      decoderStack
--                      (query', key, decoderAttentionBias, crossAttentionBias)
--              )

-- instance
--   HasForwardTransformerDecoderStack
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
--     keyEmbedDim
--     ffnDim
--     dropoutP
--     query
--     key
--     decoderAttentionBias
--     crossAttentionBias
--     generator
--     query
--     generator =>
--   HasForwardTransformerDecoderStack
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
--     keyEmbedDim
--     ffnDim
--     dropoutP
--     query
--     key
--     decoderAttentionBias
--     crossAttentionBias
--     generator
--     query
--     generator
--   where
--   forwardTransformerDecoderStack (Just f) (TransformerDecoderStackCons decoderBlock decoderStack) (query, key, decoderAttentionBias, crossAttentionBias) =
--     runIxState $
--       ireturn (query, key, decoderAttentionBias, crossAttentionBias)
--         >>>= IxState . f decoderBlock
--         >>>= ( \query' ->
--                  IxState $
--                    forwardTransformerDecoderStack
--                      @(1 <=? numLayers - 1)
--                      @'True
--                      @(numLayers - 1)
--                      (Just f)
--                      decoderStack
--                      (query', key, decoderAttentionBias, crossAttentionBias)
--              )

-- -- | 'HasForward' instance for 'TransformerDecoderStack'.
-- --
-- -- @
-- -- ┌───────┐  ┌─────┐  ┌──────────────────────┐  ┌────────────────────┐
-- -- │ query │  │ key │  │ decoderAttentionBias │  │ crossAttentionBias │
-- -- └───┬───┘  └──┬──┘  └──────────┬───────────┘  └─────────┬──────────┘
-- --     │         │                │                        │
-- --     ▼         │                │                        │
-- --  tdsBlock◄────┤◄───────────────┤◄───────────────────────┤
-- --     ▼         │                │                        │
-- --  tdsBlock◄────┤◄───────────────┤◄───────────────────────┤
-- --     ▼         │                │                        │
-- --    ...       ...              ...                      ...
-- --     ▼         │                │                        │
-- --  tdsBlock◄────┘◄───────────────┘◄───────────────────────┘
-- --     │
-- --     ▼
-- -- ┌───────┐
-- -- │ query │
-- -- └───────┘
-- -- @
-- instance
--   HasForwardTransformerDecoderStack
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
--     keyEmbedDim
--     ffnDim
--     dropoutP
--     query
--     key
--     decoderAttentionBias
--     crossAttentionBias
--     generator
--     output
--     generatorOutput =>
--   HasForward
--     (TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
--     (query, key, decoderAttentionBias, crossAttentionBias)
--     generator
--     output
--     generatorOutput
--   where
--   forward = forwardTransformerDecoderStack @(1 <=? numLayers) @'False Nothing

testDecoderStack = do
  let device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      keyEmbedDim = queryEmbedDim
      ffnDim = SName @"*" :&: SSize @2048
      dropoutP :: Float = 0.0
      eps = 1e-6
  g <- sMkGenerator device 0
  let (decoderStack, g') = initialize @(TransformerDecoderStack 2 'T5 _ _ _ _ _ _ _ _ _) (device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, ffnDim, dropoutP, eps) g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      decoderSeqDim = SName @"*" :&: SSize @13
      sOnes' = sOnes SWithoutGradient (SLayout SDense) device
      -- query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      query = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      decoderAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  let (output, _) = forward decoderStack (query, key, decoderAttentionBias, crossAttentionBias) g'
  pure output