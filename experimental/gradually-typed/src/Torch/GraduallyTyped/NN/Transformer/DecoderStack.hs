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
{-# OPTIONS_GHC -v2 -Wall -fconstraint-solver-iterations=2 #-}

module Torch.GraduallyTyped.NN.Transformer.DecoderStack where

import Data.Kind (Type)
import Data.Singletons (SingI)
import Data.Singletons.Prelude.List (SList (..))
import qualified Data.Vector as V
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import GHC.TypeLits (KnownNat, Nat, Symbol, type (+))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (TransformerDecoderBlock)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (..))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)

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
    (dropoutP :: Type)
  where
  TransformerDecoderStack ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
    VS.Vector numLayers (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) ->
    TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

instance
  ( KnownNat numLayers,
    HasInitialize
      (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      input
      generator
      generator',
    HasInitialize
      (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      input
      generator'
      generator',
    numLayers' ~ (numLayers + 1)
  ) =>
  HasInitialize
    (TransformerDecoderStack style numLayers' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    input
    generator
    generator'
  where
  initialize input g =
    let (v, g') = initialize input g
     in (TransformerDecoderStack v, g')

instance
  ( KnownNat numLayers,
    SingI style,
    HasStateDict
      (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      input
  ) =>
  HasStateDict
    (TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    input
  where
  fromStateDict input k = TransformerDecoderStack <$> fromStateDict input k
  toStateDict k (TransformerDecoderStack v) = toStateDict k v

instance
  HasForward
    (TransformerDecoderStack style 0 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    query
    generator
  where
  forward _ (query, _, _, _) = pure . (query,)

instance
  HasForward
    (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    output
    generatorOutput =>
  HasForward
    (TransformerDecoderStack style 1 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    output
    generatorOutput
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
      (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      (query, key, decoderAttentionBias, crossAttentionBias)
      generator
      output
      generatorOutput,
    HasForward
      (TransformerDecoderBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      (output, key, decoderAttentionBias, crossAttentionBias)
      generatorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    (query, key, decoderAttentionBias, crossAttentionBias)
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoderStack (VGS.Vector v)) (query, key, decoderAttentionBias, crossAttentionBias) g =
    let Just (block, blocks) = V.uncons v
     in V.foldl
          ( \agg block' -> do
              (output, g') <- agg
              forward block' (output, key, decoderAttentionBias, crossAttentionBias) g'
          )
          (forward block (query, key, decoderAttentionBias, crossAttentionBias) g)
          blocks

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
      dropoutP :: Float = 0.0
      eps = 1e-6
  g <- sMkGenerator device 0
  let (decoderStack, g') = initialize @(TransformerDecoderStack 'T5 2 _ _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, ffnDim, dropoutP, eps) g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      decoderSeqDim = SName @"*" :&: SSize @13
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      -- query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      query = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      decoderAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoderStack (query, key, decoderAttentionBias, crossAttentionBias) g'
  pure output