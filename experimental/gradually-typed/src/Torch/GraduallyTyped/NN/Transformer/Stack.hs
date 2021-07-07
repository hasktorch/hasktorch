{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverlappingInstances #-}
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
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.Stack where

import Data.Kind (Type)
import Data.Singletons (SingI)
import Data.Singletons.Prelude.List (SList (SNil))
import qualified Data.Vector as V
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import GHC.TypeLits (KnownNat, Nat, Symbol, type (+))
import Torch.GraduallyTyped.DType (DType (..), DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (SLayout), SLayoutType (SDense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.Block (TransformerBlock)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (T5))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SName (..), SShape (SShape), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)

-- | Transformer encoder stack.
newtype
  TransformerStack
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
    (dropoutP :: Type)
  where
  TransformerStack ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    VS.Vector numLayers (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) ->
    TransformerStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

instance
  ( KnownNat numLayers,
    HasInitialize
      (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      input
      generator
      generator',
    HasInitialize
      (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      input
      generator'
      generator',
    numLayers' ~ (numLayers + 1)
  ) =>
  HasInitialize
    (TransformerStack style numLayers' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    input
    generator
    generator'
  where
  initialize input g =
    let (v, g') = initialize input g
     in (TransformerStack v, g')

instance
  ( KnownNat numLayers,
    SingI style,
    HasStateDict
      (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      input
  ) =>
  HasStateDict
    (TransformerStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    input
  where
  fromStateDict input k = TransformerStack <$> fromStateDict input k
  toStateDict k (TransformerStack v) = toStateDict k v

instance
  HasForward
    (TransformerStack style 0 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    (query, attentionBias)
    generator
    query
    generator
  where
  forward _ (query, _) = pure . (query,)

instance
  HasForward
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    (query, attentionBias)
    generator
    output
    generatorOutput =>
  HasForward
    (TransformerStack style 1 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    (query, attentionBias)
    generator
    output
    generatorOutput
  where
  forward (TransformerStack (VGS.Vector v)) input g =
    let Just (block, _) = V.uncons v
     in forward block input g

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
  {-# OVERLAPPABLE #-}
  ( HasForward
      (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      (query, attentionBias)
      generator
      output
      generatorOutput,
    HasForward
      (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      (output, attentionBias)
      generatorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerStack style n gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    (query, attentionBias)
    generator
    output
    generatorOutput
  where
  forward (TransformerStack (VGS.Vector v)) (query, attentionBias) g =
    let Just (block, blocks) = V.uncons v
     in V.foldl
          ( \agg block' -> do
              (output, g') <- agg
              forward block' (output, attentionBias) g'
          )
          (forward block (query, attentionBias) g)
          blocks

testStack = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      dropoutP :: Float = 0.0
      eps = 1e-6
  g <- sMkGenerator device 0
  let (stack, g') = initialize @(TransformerStack 'T5 2 _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, ffnDim, dropoutP, eps) g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward stack (query, attentionBias) g'
  pure output
