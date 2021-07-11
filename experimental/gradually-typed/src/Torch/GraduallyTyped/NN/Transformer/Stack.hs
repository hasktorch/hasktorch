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
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.Stack where

import Control.Monad.Indexed (IxPointed (..), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Singletons (SingI)
import Data.Singletons.Prelude.List (SList (SNil))
import Data.Singletons.TypeLits (SNat (..))
import qualified Data.Vector as V
import qualified Data.Vector.Generic.Sized.Internal as VGS
import qualified Data.Vector.Sized as VS
import GHC.TypeLits (Nat, Symbol, type (+))
import Torch.GraduallyTyped.DType (DType (..), DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (SLayout), SLayoutType (SDense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, VectorSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Block (TransformerBlock, TransformerBlockSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (ST5), TransformerStyle)
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (SShape), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (TensorSpec (TensorSpec))

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
  where
  TransformerStack ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim.
    VS.Vector numLayers (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim) ->
    TransformerStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim

data
  TransformerStackSpec
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
  where
  TransformerStackSpec ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim.
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
    Double ->
    Double ->
    TransformerStackSpec style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim

type instance ModelSpec (TransformerStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim) = TransformerStackSpec style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim

instance
  ( block ~ TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim,
    HasInitialize block device block device,
    numLayers' ~ (numLayers + 1)
  ) =>
  HasInitialize
    (TransformerStack style numLayers' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    generatorDevice
    (TransformerStack style numLayers' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    device
  where
  initialize (TransformerStackSpec style numLayers'@SNat gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        blockSpec = TransformerBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps
        v = IxStateT . initialize @(VS.Vector numLayers' block) $ VectorSpec numLayers' (VS.replicate' numLayers' blockSpec)
     in runIxStateT (v >>>= ireturn . TransformerStack) generator'

instance
  ( SingI style,
    HasStateDict
      (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
  ) =>
  HasStateDict
    (TransformerStack style numLayers gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
  where
  fromStateDict (TransformerStackSpec style numLayers'@SNat gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps) k =
    let blockSpec = TransformerBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps
     in TransformerStack <$> fromStateDict (VectorSpec numLayers' $ VS.replicate' numLayers' blockSpec) k
  toStateDict k (TransformerStack v) = toStateDict k v

instance
  HasForward
    (TransformerStack style 0 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    (query, attentionBias)
    generator
    query
    generator
  where
  forward _ (query, _) = pure . (query,)

instance
  HasForward
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice =>
  HasForward
    (TransformerStack style 1 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
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
      (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
      (query, attentionBias)
      generatorDevice
      output
      generatorOutputDevice,
    HasForward
      (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
      (output, attentionBias)
      generatorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerStack style n gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
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

testStack :: IO _
testStack = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      dropoutP = 0.0
      eps = 1e-6
  let g = sMkGenerator device 0
  (stack, g') <- initialize (TransformerStackSpec ST5 (SNat @2) gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps) g
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @17
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward stack (query, attentionBias) g'
  pure output
