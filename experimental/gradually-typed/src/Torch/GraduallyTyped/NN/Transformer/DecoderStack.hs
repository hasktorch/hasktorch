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
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -v2 -Wall -fconstraint-solver-iterations=3 #-}

module Torch.GraduallyTyped.NN.Transformer.DecoderStack where

import Control.Monad.Indexed (IxPointed (..), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Singletons (SingI)
import Data.Singletons.Prelude.List (SList (..))
import Data.Singletons.TypeLits (SNat (SNat))
import qualified Data.Vector as V
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import Debug.Trace (traceShow)
import GHC.TypeLits (KnownNat, Nat, Symbol, type (+))
import Torch.GraduallyTyped.DType (DType (..), DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, VectorSpec (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (TransformerDecoderBlock, TransformerDecoderBlockSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (ST5), TransformerStyle (..))
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (TensorSpec (TensorSpec))

-- | Transformer decoder stack.
data
  TransformerDecoderStack
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
  where
  TransformerDecoderStack ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim.
    VS.Vector numLayers (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim) ->
    TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim

data
  TransformerDecoderStackSpec
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
  where
  TransformerDecoderStackSpec ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim.
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
    Double ->
    Double ->
    TransformerDecoderStackSpec style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim

type instance ModelSpec (TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim) = TransformerDecoderStackSpec style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim

instance
  ( decoderBlock ~ TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim,
    HasInitialize decoderBlock device decoderBlock device,
    numLayers' ~ (numLayers + 1)
  ) =>
  HasInitialize
    (TransformerDecoderStack style numLayers' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    generatorDevice
    (TransformerDecoderStack style numLayers' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    device
  where
  initialize (TransformerDecoderStackSpec style numLayers'@SNat gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        decoderBlockSpec = TransformerDecoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps
        v = IxStateT . initialize @(VS.Vector numLayers' decoderBlock) $ VectorSpec numLayers' (VS.replicate' numLayers' decoderBlockSpec)
     in runIxStateT (v >>>= ireturn . TransformerDecoderStack) generator'

instance
  ( SingI style,
    HasStateDict
      (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
  ) =>
  HasStateDict
    (TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
  where
  fromStateDict (TransformerDecoderStackSpec style numLayers'@SNat gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps) k =
    let decoderBlockSpec = TransformerDecoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps
     in TransformerDecoderStack <$> fromStateDict (VectorSpec numLayers' $ VS.replicate' numLayers' decoderBlockSpec) k
  toStateDict k (TransformerDecoderStack v) = toStateDict k v

instance
  HasForward
    (TransformerDecoderStack style 0 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    query
    generator
  where
  forward _ (query, _, _, _) = pure . (query,)

instance
  HasForward
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generatorDevice
    output
    generatorOutputDevice =>
  HasForward
    (TransformerDecoderStack style 1 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerDecoderStack (VGS.Vector v)) input g =
    let Just (block, _) = V.uncons v
     in forward block input g

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
  {-# OVERLAPPABLE #-}
  ( HasForward
      (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
      (query, key, decoderAttentionBias, crossAttentionBias)
      generatorDevice
      output
      generatorOutputDevice,
    HasForward
      (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
      (output, key, decoderAttentionBias, crossAttentionBias)
      generatorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerDecoderStack (VGS.Vector v)) (query, key, decoderAttentionBias, crossAttentionBias) g =
    let Just (block, blocks) = V.uncons v
     in V.foldl
          ( \agg block' -> do
              (output, g') <- agg
              (output', g'') <- forward block' (output, key, decoderAttentionBias, crossAttentionBias) g'
              pure (output', g'')
          )
          ( do
              (output, g') <- forward block (query, key, decoderAttentionBias, crossAttentionBias) g
              pure (output, g')
          )
          blocks

testDecoderStack :: IO _
testDecoderStack = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      keyEmbedDim = queryEmbedDim
      ffnDim = SName @"*" :&: SSize @2048
      dropoutP = 0.0
      eps = 1e-6
  let g = sMkGenerator device 0
  (decoderStack, g') <- initialize (TransformerDecoderStackSpec ST5 (SNat @2) gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps) g
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      decoderSeqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      decoderAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoderStack (query, key, decoderAttentionBias, crossAttentionBias) g'
  pure output