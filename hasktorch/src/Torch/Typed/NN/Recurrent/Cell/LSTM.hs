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

module Torch.Typed.NN.Recurrent.Cell.LSTM where

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

-- | A specification for a long, short-term memory (LSTM) cell.
data
  LSTMCellSpec
    (inputDim :: Nat)
    (hiddenDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = -- | Weights and biases are drawn from the standard normal distibution (having mean 0 and variance 1)
    LSTMCellSpec
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)

-- | A long, short-term memory cell.
data
  LSTMCell
    (inputDim :: Nat)
    (hiddenDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) = LSTMCell
  { -- | input-to-hidden weights
    lstmCell_w_ih :: Parameter device dtype '[4 * hiddenDim, inputDim],
    -- | hidden-to-hidden weights
    lstmCell_w_hh :: Parameter device dtype '[4 * hiddenDim, hiddenDim],
    -- | input-to-hidden bias
    lstmCell_b_ih :: Parameter device dtype '[4 * hiddenDim],
    -- | hidden-to-hidden bias
    lstmCell_b_hh :: Parameter device dtype '[4 * hiddenDim]
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
    (LSTMCellSpec inputDim hiddenDim dtype device)
    (LSTMCell inputDim hiddenDim dtype device)
  where
  sample LSTMCellSpec =
    LSTMCell
      <$> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)

-- | A single recurrent step of an `LSTMCell`
lstmCellForward ::
  forall inputDim hiddenDim batchSize dtype device.
  ( KnownDType dtype,
    KnownNat inputDim,
    KnownNat hiddenDim,
    KnownNat batchSize
  ) =>
  -- | The cell
  LSTMCell inputDim hiddenDim dtype device ->
  -- | The current (Hidden, Cell) state
  ( Tensor device dtype '[batchSize, hiddenDim],
    Tensor device dtype '[batchSize, hiddenDim]
  ) ->
  -- | The input
  Tensor device dtype '[batchSize, inputDim] ->
  -- | The subsequent (Hidden, Cell) state
  ( Tensor device dtype '[batchSize, hiddenDim],
    Tensor device dtype '[batchSize, hiddenDim]
  )
lstmCellForward LSTMCell {..} =
  lstmCell
    (toDependent lstmCell_w_ih)
    (toDependent lstmCell_w_hh)
    (toDependent lstmCell_b_ih)
    (toDependent lstmCell_b_hh)

-- | foldl' for lists of tensors unsing an `LSTMCell`
lstmCellFold ::
  forall inputDim hiddenDim batchSize dtype device.
  ( KnownDType dtype,
    KnownNat inputDim,
    KnownNat hiddenDim,
    KnownNat batchSize
  ) =>
  LSTMCell inputDim hiddenDim dtype device ->
  -- | The initial (Hidden, Cell) state
  ( Tensor device dtype '[batchSize, hiddenDim],
    Tensor device dtype '[batchSize, hiddenDim]
  ) ->
  -- | The list of inputs
  [Tensor device dtype '[batchSize, inputDim]] ->
  -- | The final (Hidden, Cell) state
  ( Tensor device dtype '[batchSize, hiddenDim],
    Tensor device dtype '[batchSize, hiddenDim]
  )
lstmCellFold cell = foldl' (lstmCellForward cell)

-- | scanl' for lists of tensors unsing an `LSTMCell`
lstmCellScan ::
  forall inputDim hiddenDim batchSize dtype device.
  ( KnownDType dtype,
    KnownNat inputDim,
    KnownNat hiddenDim,
    KnownNat batchSize
  ) =>
  LSTMCell inputDim hiddenDim dtype device ->
  -- | The initial (Hidden, Cell) state
  ( Tensor device dtype '[batchSize, hiddenDim],
    Tensor device dtype '[batchSize, hiddenDim]
  ) ->
  -- | The list of inputs
  [Tensor device dtype '[batchSize, inputDim]] ->
  -- | All subsequent (Hidden, Cell) states
  [ ( Tensor device dtype '[batchSize, hiddenDim],
      Tensor device dtype '[batchSize, hiddenDim]
    )
  ]
lstmCellScan cell = scanl' (lstmCellForward cell)
