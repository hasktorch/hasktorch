{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.GStack where

import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>))
import Data.Kind (Type)
import qualified Data.Vector as V hiding (uncons)
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol, type (+))
import Torch.GraduallyTyped.DType (DType, DataType, SDataType (..))
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice (..))
import qualified Torch.GraduallyTyped.Internal.Vector as V
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, VectorSpec (..))
import Torch.GraduallyTyped.NN.Transformer.GBlock (DecoderBlockF, EncoderBlockF, decoderBlockSpec, encoderBlockSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle, TransformerStyle)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude.TypeLits
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim, Name, SDim, Size)

-- | Generic transformer stack.
--
-- - @stack@ is a stack of tranformer blocks.
newtype GTransformerStack (stack :: Type) where
  GTransformerStack :: forall stack. stack -> GTransformerStack stack
  deriving stock (Eq, Ord, Show, Generic)

type instance
  ModelSpec (GTransformerStack stack) =
    GTransformerStack (ModelSpec stack)

type family
  EncoderStackF
    (style :: TransformerStyle)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (hasDropout :: HasDropout)
  where
  EncoderStackF style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim hasDropout =
    GTransformerStack
      ( VS.Vector
          numLayers
          (EncoderBlockF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim hasDropout)
      )

-- | Specifies the parameters of a transformer stack in an encoder configuration.
--
-- - @style@: the style of the transformer stack, e.g. 'ST5', 'SByT5', etc.
-- - @gradient@: whether to compute the gradient of the stack's parameters.
-- - @device@: the computational device on which the stack is allocated.
-- - @dataType@: the data type of the stack's parameters.
-- - @headDim@: the dimension of all transformer heads in the stack.
-- - @headEmbedDim@: the dimension of the transformer head embeddings.
-- - @embedDim@: the dimension of the transformer embeddings.
-- - @queryEmbedDim@: the dimension of the transformer query embeddings.
-- - @ffnDim@: the dimension of the feed-forward network.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
encoderStackSpec ::
  forall style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim hasDropout.
  STransformerStyle style ->
  SNat numLayers ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim queryEmbedDim ->
  SDim ffnDim ->
  SHasDropout hasDropout ->
  Double ->
  Double ->
  ModelSpec (EncoderStackF style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim hasDropout)
encoderStackSpec style numLayers@SNat gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim hasDropout dropoutP eps =
  let blockSpec = encoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim hasDropout dropoutP eps
   in GTransformerStack $ VectorSpec numLayers (VS.replicate' numLayers blockSpec)

type family
  DecoderStackF
    (style :: TransformerStyle)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (hasDropout :: HasDropout)
  where
  DecoderStackF style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim hasDropout =
    GTransformerStack
      ( VS.Vector
          numLayers
          (DecoderBlockF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim hasDropout)
      )

-- | Specifies the parameters of a transformer stack in a decoder configuration.
--
-- - @style@: the style of the transformer stack, e.g. 'ST5', 'SByT5', etc.
-- - @gradient@: whether to compute the gradient of the stack's parameters.
-- - @device@: the computational device on which the stack is allocated.
-- - @dataType@: the data type of the stack's parameters.
-- - @headDim@: the dimension of all transformer heads in the stack.
-- - @headEmbedDim@: the dimension of the transformer head embeddings.
-- - @embedDim@: the dimension of the transformer embeddings.
-- - @queryEmbedDim@: the dimension of the transformer query embeddings.
-- - @keyEmbedDim@: the dimension of the transformer key embeddings.
-- - @ffnDim@: the dimension of the feed-forward network.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
decoderStackSpec ::
  forall style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim hasDropout.
  STransformerStyle style ->
  SNat numLayers ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim queryEmbedDim ->
  SDim keyEmbedDim ->
  SDim ffnDim ->
  SHasDropout hasDropout ->
  Double ->
  Double ->
  ModelSpec (DecoderStackF style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim hasDropout)
decoderStackSpec style numLayers@SNat gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim hasDropout dropoutP eps =
  let blockSpec = decoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim hasDropout dropoutP eps
   in GTransformerStack $ VectorSpec numLayers (VS.replicate' numLayers blockSpec)

instance
  ( HasInitialize block generatorDevice block' generatorDevice,
    numLayers' ~ (numLayers + 1)
  ) =>
  HasInitialize
    (GTransformerStack (VS.Vector numLayers' block))
    generatorDevice
    (GTransformerStack (VS.Vector numLayers' block'))
    generatorDevice

instance
  HasStateDict block =>
  HasStateDict (GTransformerStack (VS.Vector numLayers block))

instance
  HasForward
    (GTransformerStack (VS.Vector 0 block))
    (query, attentionBias)
    generatorDevice
    query
    generatorDevice
  where
  forward _ (query, _) = pure . (query,)

instance
  HasForward
    block
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice =>
  HasForward
    (GTransformerStack (VS.Vector 1 block))
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (GTransformerStack (VGS.Vector v)) input g =
    let Just (block, _) = V.uncons v
     in forward block input g

instance
  HasForward
    (GTransformerStack (VS.Vector 0 block))
    (query, key, attentionBias, crossAttentionBias)
    generator
    query
    generator
  where
  forward _ (query, _, _, _) = pure . (query,)

instance
  HasForward
    block
    (query, key, attentionBias, crossAttentionBias)
    generatorDevice
    output
    generatorOutputDevice =>
  HasForward
    (GTransformerStack (VS.Vector 1 block))
    (query, key, attentionBias, crossAttentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (GTransformerStack (VGS.Vector v)) input g =
    let Just (block, _) = V.uncons v
     in forward block input g

-- | 'HasForward' instance for 'GTransformerStack' in an encoder configuration.
--
-- @
-- ┌───────┐  ┌───────────────┐
-- │ query │  │ attentionBias │
-- └───┬───┘  └───────┬───────┘
--     │              │
--     ▼              │
--   block◄───────────┤
--     ▼              │
--   block◄───────────┤
--     ▼              │
--    ...            ...
--     ▼              │
--   block◄───────────┘
--     │
--     ▼
-- ┌───────┐
-- │ query │
-- └───────┘
-- @
instance
  {-# OVERLAPPABLE #-}
  ( HasForward
      block
      (query, attentionBias)
      generatorDevice
      output
      generatorOutputDevice,
    HasForward
      block
      (output, attentionBias)
      generatorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GTransformerStack (VS.Vector n block))
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (GTransformerStack (VGS.Vector v)) (query, attentionBias) g =
    let Just (block, blocks) = V.uncons v
     in V.foldl
          ( \agg block' -> do
              (output, g') <- agg
              (output', g'') <- forward block' (output, attentionBias) g'
              pure (output', g'')
          )
          ( do
              (output, g') <- forward block (query, attentionBias) g
              pure (output, g')
          )
          blocks

-- | 'HasForward' instance for 'GTransformerStack' in a decoder configuration.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐  ┌────────────────────┐
-- │ query │  │ key │  │ attentionBias │  │ crossAttentionBias │
-- └───┬───┘  └──┬──┘  └───────┬───────┘  └─────────┬──────────┘
--     │         │             │                    │
--     ▼         │             │                    │
--   block◄──────┤◄────────────┤◄───────────────────┤
--     ▼         │             │                    │
--   block◄──────┤◄────────────┤◄───────────────────┤
--     ▼         │             │                    │
--    ...       ...           ...                  ...
--     ▼         │             │                    │
--   block◄──────┘◄────────────┘◄───────────────────┘
--     │
--     ▼
-- ┌───────┐
-- │ query │
-- └───────┘
-- @
instance
  {-# OVERLAPPABLE #-}
  ( HasForward
      block
      (query, key, attentionBias, crossAttentionBias)
      generatorDevice
      output
      generatorOutputDevice,
    HasForward
      block
      (output, key, attentionBias, crossAttentionBias)
      generatorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GTransformerStack (VS.Vector n block))
    (query, key, attentionBias, crossAttentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (GTransformerStack (VGS.Vector v)) (query, key, attentionBias, crossAttentionBias) g =
    let Just (block, blocks) = V.uncons v
     in V.foldl
          ( \agg block' -> do
              (output, g') <- agg
              (output', g'') <- forward block' (output, key, attentionBias, crossAttentionBias) g'
              pure (output', g'')
          )
          ( do
              (output, g') <- forward block (query, key, attentionBias, crossAttentionBias) g
              pure (output, g')
          )
          blocks
