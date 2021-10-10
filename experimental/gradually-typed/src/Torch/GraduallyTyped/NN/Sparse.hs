{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Sparse where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Data (Proxy (..))
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat, Symbol, natVal)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType, SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF, embedding)
import Torch.GraduallyTyped.Prelude (Catch, pattern (:|:))
import Torch.GraduallyTyped.Prelude.List (SList (..))
import Torch.GraduallyTyped.Prelude.Maybe (SMaybe (..))
import Torch.GraduallyTyped.Random (SGetGeneratorDevice)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name, SDim (..), SShape (..), Shape (..), Size)
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
  deriving stock (Show, Generic)

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
  deriving stock (Show, Generic)

type instance ModelSpec (Embedding gradient layout device dataType embedNumDim embedDim paddingIdx) = EmbeddingSpec gradient layout device dataType embedNumDim embedDim paddingIdx

instance
  ( output ~ Embedding gradient layout (device <+> generatorDevice) dataType embedNumDim embedDim paddingIdx,
    generatorOutputDevice ~ (device <+> generatorDevice),
    SGetGeneratorDevice generatorDevice
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
  fromStateDict (EmbeddingSpec gradient layout device dataType embedNumDim embedDim _paddingIdx) k =
    Embedding <$> fromStateDict (TensorSpec gradient layout device dataType (SShape $ embedNumDim :|: embedDim :|: SNil)) (k <> "weight")
  toStateDict k Embedding {..} =
    toStateDict (k <> "weight") embeddingWeight

instance
  ( SGetLayout layout,
    Catch (dataType' <+> 'DataType 'Int64),
    output
      ~ Tensor
          (gradient <|> gradient')
          (layout <+> layout')
          (device <+> device')
          dataType
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
    Catch (dataType' <+> 'DataType 'Int64),
    output
      ~ Tensor
          (gradient <|> gradient')
          (layout <+> layout')
          (device <+> device')
          dataType
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
