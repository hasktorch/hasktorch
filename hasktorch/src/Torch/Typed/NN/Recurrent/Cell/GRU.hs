{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module Torch.Typed.NN.Recurrent.Cell.GRU where

import Data.List
  ( foldl',
    scanl',
  )
import GHC.Generics
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.NN as A
import Torch.Typed.Factories
import Torch.Typed.Functional hiding (linear)
import Torch.Typed.NN.Dropout
import Torch.Typed.Parameter
import Torch.Typed.Tensor

-- | A specification for a gated recurrent unit (GRU) cell.
data
  GRUCellSpec
    (inputDim :: Nat)
    (hiddenDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = -- | Weights and biases are drawn from the standard normal distibution (having mean 0 and variance 1)
    GRUCellSpec
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)

-- | A gated recurrent unit (GRU) cell.
data
  GRUCell
    (inputDim :: Nat)
    (hiddenDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) = GRUCell
  { -- | input-to-hidden weights
    gruCell_w_ih :: Parameter device dtype '[3 * hiddenDim, inputDim],
    -- | hidden-to-hidden weights
    gruCell_w_hh :: Parameter device dtype '[3 * hiddenDim, hiddenDim],
    -- | input-to-hidden bias
    gruCell_b_ih :: Parameter device dtype '[3 * hiddenDim],
    -- | hidden-to-hidden bias
    gruCell_b_hh :: Parameter device dtype '[3 * hiddenDim]
  }
  deriving (Show, Generic, Parameterized)

instance
  ( KnownDevice device,
    KnownDType dtype,
    KnownNat inputDim,
    KnownNat hiddenDim,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (GRUCellSpec inputDim hiddenDim dtype device)
    (GRUCell inputDim hiddenDim dtype device)
  where
  sample GRUCellSpec =
    GRUCell
      <$> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)

-- | A single recurrent step of a `GRUCell`
gruCellForward ::
  forall inputDim hiddenDim batchSize dtype device.
  ( KnownDType dtype,
    KnownNat inputDim,
    KnownNat hiddenDim,
    KnownNat batchSize
  ) =>
  -- | The cell
  GRUCell inputDim hiddenDim dtype device ->
  -- | The current Hidden state
  Tensor device dtype '[batchSize, hiddenDim] ->
  -- | The input
  Tensor device dtype '[batchSize, inputDim] ->
  -- | The subsequent Hidden state
  Tensor device dtype '[batchSize, hiddenDim]
gruCellForward GRUCell {..} =
  gruCell
    (toDependent gruCell_w_ih)
    (toDependent gruCell_w_hh)
    (toDependent gruCell_b_ih)
    (toDependent gruCell_b_hh)

-- | foldl' for lists of tensors unsing a `GRUCell`
gruFold ::
  forall inputDim hiddenDim batchSize dtype device.
  ( KnownDType dtype,
    KnownNat inputDim,
    KnownNat hiddenDim,
    KnownNat batchSize
  ) =>
  GRUCell inputDim hiddenDim dtype device ->
  -- | The initial Hidden state
  Tensor device dtype '[batchSize, hiddenDim] ->
  -- | The list of inputs
  [Tensor device dtype '[batchSize, inputDim]] ->
  -- | The final Hidden state
  Tensor device dtype '[batchSize, hiddenDim]
gruFold cell = foldl' (gruCellForward cell)

-- | scanl' for lists of tensors unsing a `GRUCell`
gruCellScan ::
  forall inputDim hiddenDim batchSize dtype device.
  ( KnownDType dtype,
    KnownNat inputDim,
    KnownNat hiddenDim,
    KnownNat batchSize
  ) =>
  GRUCell inputDim hiddenDim dtype device ->
  -- | The initial Hidden state
  Tensor device dtype '[batchSize, hiddenDim] ->
  -- | The list of inputs
  [Tensor device dtype '[batchSize, inputDim]] ->
  -- | All subsequent Hidden states
  [Tensor device dtype '[batchSize, hiddenDim]]
gruCellScan cell = scanl' (gruCellForward cell)
