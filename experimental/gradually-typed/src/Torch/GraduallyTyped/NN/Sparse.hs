{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Sparse where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Data (Proxy (..))
import Data.Singletons.Prelude.List (SList (..))
import Data.Singletons.Prelude.Maybe (SMaybe (..))
import GHC.TypeLits (KnownNat, Nat, Symbol, natVal)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType, SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF, embedding)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name, SDim (..), SShape (..), Shape (..), Size, pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sRandn)
import Torch.GraduallyTyped.Tensor.Type (SGetLayout, Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

data
  Embedding
    (gradient :: Gradient RequiresGradient)
    (layout :: Layout LayoutType)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedNumDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (paddingIdx :: Maybe Nat)
  where
  Embedding ::
    forall gradient layout device dataType embedNumDim embedDim paddingIdx.
    { embeddingWeight :: Tensor gradient layout device dataType ('Shape '[embedNumDim, embedDim])
    } ->
    Embedding gradient layout device dataType embedNumDim embedDim paddingIdx

data
  EmbeddingSpec
    (gradient :: Gradient RequiresGradient)
    (layout :: Layout LayoutType)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedNumDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (paddingIdx :: Maybe Nat)
  where
  EmbeddingSpec ::
    forall gradient layout device dataType embedNumDim embedDim paddingIdx.
    SGradient gradient ->
    SLayout layout ->
    SDevice device ->
    SDataType dataType ->
    SDim embedNumDim ->
    SDim embedDim ->
    SMaybe paddingIdx ->
    EmbeddingSpec gradient layout device dataType embedNumDim embedDim paddingIdx

type instance ModelSpec (Embedding gradient layout device dataType embedNumDim embedDim paddingIdx) = EmbeddingSpec gradient layout device dataType embedNumDim embedDim paddingIdx

instance
  ( output ~ Embedding gradient layout (device <+> generatorDevice) dataType embedNumDim embedDim paddingIdx,
    generatorOutputDevice ~ (device <+> generatorDevice)
  ) =>
  HasInitialize
    (Embedding gradient layout device dataType embedNumDim embedDim paddingIdx)
    generatorDevice
    output
    generatorOutputDevice
  where
  initialize (EmbeddingSpec gradient layout device dataType embedNumDim embedDim SNothing) =
    runIxStateT $
      IxStateT (sRandn $ TensorSpec gradient layout device dataType (SShape $ embedNumDim :|: embedDim :|: SNil))
        >>>= ireturn . Embedding
  initialize (EmbeddingSpec gradient layout device dataType embedNumDim embedDim (SJust _)) =
    -- TODO: padding embedding vector may need to be set to zeros
    runIxStateT $
      IxStateT (sRandn $ TensorSpec gradient layout device dataType (SShape $ embedNumDim :|: embedDim :|: SNil))
        >>>= ireturn . Embedding

instance
  HasStateDict
    (Embedding gradient layout device dataType embedNumDim embedDim paddingIdx)
  where
  fromStateDict (EmbeddingSpec gradient layout device dataType embedNumDim embedDim paddingIdx) k =
    Embedding <$> fromStateDict (TensorSpec gradient layout device dataType (SShape $ embedNumDim :|: embedDim :|: SNil)) (k <> "weight")
  toStateDict k Embedding {..} =
    toStateDict (k <> "weight") embeddingWeight

instance
  ( SGetLayout layout,
    output
      ~ Tensor
          (gradient <|> gradient')
          (layout <+> layout')
          (device <+> device')
          (Seq (dataType' <+> 'DataType 'Int64) dataType)
          (EmbeddingF ('Shape '[embedNumDim, embedDim]) shape')
  ) =>
  HasForward
    (Embedding gradient layout device dataType embedNumDim embedDim 'Nothing)
    (Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorDevice
  where
  forward (Embedding weight) input = pure . (embedding Nothing False weight input,)

instance
  ( SGetLayout layout,
    KnownNat paddingIdx,
    output
      ~ Tensor
          (gradient <|> gradient')
          (layout <+> layout')
          (device <+> device')
          (Seq (dataType' <+> 'DataType 'Int64) dataType)
          (EmbeddingF ('Shape '[embedNumDim, embedDim]) shape')
  ) =>
  HasForward
    (Embedding gradient layout device dataType embedNumDim embedDim ('Just paddingIdx))
    (Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorDevice
  where
  forward Embedding {..} input = pure . (embedding (Just . fromIntegral . natVal $ Proxy @paddingIdx) False embeddingWeight input,)
