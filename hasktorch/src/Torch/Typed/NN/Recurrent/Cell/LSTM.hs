{-# LANGUAGE AllowAmbiguousTypes     #-}
{-# LANGUAGE DataKinds               #-}
{-# LANGUAGE DeriveGeneric           #-}
{-# LANGUAGE FlexibleContexts        #-}
{-# LANGUAGE FlexibleInstances       #-}
{-# LANGUAGE MultiParamTypeClasses   #-}
{-# LANGUAGE NoStarIsType            #-}
{-# LANGUAGE OverloadedLists         #-}
{-# LANGUAGE PolyKinds               #-}
{-# LANGUAGE RankNTypes              #-}
{-# LANGUAGE RecordWildCards         #-}
{-# LANGUAGE ScopedTypeVariables     #-}
{-# LANGUAGE TypeApplications        #-}
{-# LANGUAGE TypeFamilies            #-}
{-# LANGUAGE TypeOperators           #-}
{-# LANGUAGE UndecidableInstances    #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module Torch.Typed.NN.Recurrent.Cell.LSTM where

import           Data.List                      ( foldl'
                                                , scanl'
                                                )
import           GHC.Generics
import           GHC.TypeLits
import qualified Torch.DType                   as D
import qualified Torch.NN                      as A
import           Torch.Typed
import           Torch.Typed.Factories
import           Torch.Typed.Native      hiding ( linear )
import           Torch.Typed.NN


-- | A specification for a long, short-term memory cell.
--
data LSTMCellSpec (dtype :: D.DType) (inputDim :: Nat) (hiddenDim :: Nat) =
    LSTMCellSpec -- ^ Weights and biases are drawn from the standard normal distibution (having mean 0 and variance 1)
    deriving (Show, Eq, Ord, Generic, Enum, Bounded)

-- | A long, short-term memory cell.
--
data LSTMCell (dtype :: D.DType) (inputDim :: Nat) (hiddenDim :: Nat) =  LSTMCell {
      lstmCell_w_ih :: Parameter dtype '[4 * hiddenDim, inputDim] -- ^ input-to-hidden weights
    , lstmCell_w_hh :: Parameter dtype '[4 * hiddenDim, hiddenDim] -- ^ hidden-to-hidden weights
    , lstmCell_b_ih :: Parameter dtype '[4 * hiddenDim] -- ^ input-to-hidden bias
    , lstmCell_b_hh :: Parameter dtype '[4 * hiddenDim] -- ^ hidden-to-hidden bias
    } deriving Generic

instance (KnownDType dtype, KnownNat inputDim, KnownNat hiddenDim) => A.Randomizable (LSTMCellSpec dtype inputDim hiddenDim) (LSTMCell dtype inputDim hiddenDim) where
    sample LSTMCellSpec =
        LSTMCell
            <$> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)

instance A.Parameterized (LSTMCell d i h)

-- | A single recurrent step of an `LSTMCell`
--
forwardStep
    :: forall d i h b
     . (KnownDType d, KnownNat i, KnownNat h, KnownNat b)
    => LSTMCell d i h -- ^ The cell
    -> Tensor d '[b, i] -- ^ The input
    -> (Tensor d '[b, h], Tensor d '[b, h]) -- ^ The current (Hidden, Cell) state
    -> (Tensor d '[b, h], Tensor d '[b, h]) -- ^ The subequent (Hidden, Cell) state
forwardStep LSTMCell {..} input (hs, cs) = lstm_cell
    input
    [hs, cs]
    (toDependent lstmCell_w_ih)
    (toDependent lstmCell_w_hh)
    (toDependent lstmCell_b_ih)
    (toDependent lstmCell_b_hh)

-- | foldl' for lists of tensors unsing an `LSTMCell`
--
forward
    :: (KnownDType d, KnownNat i, KnownNat h, KnownNat b)
    => LSTMCell d i h
    -> (Tensor d '[b, h], Tensor d '[b, h])
    -> [Tensor d '[b, i]]
    -> (Tensor d '[b, h], Tensor d '[b, h])
forward lc st inputs = foldl' (\acc i -> forwardStep lc i acc) st inputs

-- | scanl' for lists of tensors unsing an `LSTMCell`
--
forwardScan
    :: (KnownDType d, KnownNat i, KnownNat h, KnownNat b)
    => LSTMCell d i h
    -> (Tensor d '[b, h], Tensor d '[b, h])
    -> [Tensor d '[b, i]]
    -> [(Tensor d '[b, h], Tensor d '[b, h])]
forwardScan lc st inputs = scanl' (\acc i -> forwardStep lc i acc) st inputs
