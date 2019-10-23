{-# LANGUAGE AllowAmbiguousTypes     #-}
{-# LANGUAGE ConstraintKinds         #-}
{-# LANGUAGE DataKinds               #-}
{-# LANGUAGE DeriveGeneric           #-}
{-# LANGUAGE FlexibleContexts        #-}
{-# LANGUAGE FlexibleInstances       #-}
{-# LANGUAGE GADTs                   #-}
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

module Torch.Typed.NN.Recurrent.LSTM where

import           Control.Monad                  ( foldM
                                                , when
                                                , void
                                                )
import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           Data.List                      ( foldl'
                                                , intersperse
                                                , scanl'
                                                )
import           Data.Maybe                     ( catMaybes )
import           Data.Reflection
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe
import           Prelude                 hiding ( tanh )
import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import qualified Torch.Autograd                as A
import qualified Torch.DType                   as D
import qualified Torch.Functions               as D
import qualified Torch.NN                      as A
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import           Torch.Typed
import           Torch.Typed.Factories
import           Torch.Typed.Native      hiding ( linear )
import           Torch.Typed.NN



data LSTMCellSpec (dtype :: D.DType) (inputDim :: Nat) (hiddenDim :: Nat) =
    LSTMCellRandomInitState -- ^ Random initial memory cells and hiddenstate values
    | LSTMCellZeroInitState -- ^ Zero-valued initial memory cells and hiddenstate values
    deriving (Show, Eq, Ord, Generic, Enum, Bounded)

data LSTMCell (dtype :: D.DType) (inputDim :: Nat) (hiddenDim :: Nat) =  LSTMCell {
    lSTMCell_init_h :: Parameter dtype '[hiddenDim] -- ^ initial hidden state
    , lSTMCell_init_c :: Parameter dtype '[hiddenDim] -- ^ initial cell state
    , lSTMCell_w_ih  :: Parameter dtype '[4 * hiddenDim, inputDim]  -- ^ input-to-hidden weights
    , lSTMCell_w_hh :: Parameter dtype '[4 * hiddenDim, hiddenDim] -- ^ hidden-to-hidden weights
    , lSTMCell_b_ih :: Parameter dtype '[4 * hiddenDim] -- ^ input-to-hidden bias
    , lSTMCell_b_hh :: Parameter dtype '[4 * hiddenDim] -- ^ hidden-to-hidden bias
    } deriving Generic

instance (KnownDType dtype, KnownNat inputDim, KnownNat hiddenDim) => A.Randomizable (LSTMCellSpec dtype inputDim hiddenDim) (LSTMCell dtype inputDim hiddenDim) where
    sample LSTMCellRandomInitState =
        LSTMCell
            <$> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
    sample LSTMCellZeroInitState =
        LSTMCell
            <$> (makeIndependent =<< pure zeros)
            <*> (makeIndependent =<< pure zeros)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)
            <*> (makeIndependent =<< randn)

instance A.Parameterized (LSTMCell d i h)

-- | A single recurrent step
-- 
forwardStep
    :: forall d i h b
     . (KnownDType d, KnownNat i, KnownNat h, KnownNat b)
    => LSTMCell d i h -- ^ The cell
    -> Tensor d '[b, i] -- ^ The input
    -> Maybe (Tensor d '[b, h], Tensor d '[b, h]) -- ^ The current (Hidden, Cell) state
    -> (Tensor d '[b, h], Tensor d '[b, h]) -- ^ The subequent (Hidden, Cell) state
forwardStep LSTMCell {..} input Nothing = lstm_cell
    input
    [ expand @'[b, h] False $ toDependent lSTMCell_init_h
    , expand @'[b, h] False $ toDependent lSTMCell_init_c
    ]
    (toDependent lSTMCell_w_ih)
    (toDependent lSTMCell_w_hh)
    (toDependent lSTMCell_b_ih)
    (toDependent lSTMCell_b_hh)

forwardStep LSTMCell {..} input (Just (hs, cs)) = lstm_cell
    input
    [hs, cs]
    (toDependent lSTMCell_w_ih)
    (toDependent lSTMCell_w_hh)
    (toDependent lSTMCell_b_ih)
    (toDependent lSTMCell_b_hh)

-- TODO: should we alias these functions with their Prelude ancesters?

-- | foldl' for lists of tensors unsing an `LSTMCell`
--
forward
    :: (KnownDType d, KnownNat i, KnownNat h, KnownNat b)
    => LSTMCell d i h
    -> [Tensor d '[b, i]]
    -> Maybe (Tensor d '[b, h], Tensor d '[b, h])
forward lc inputs =
    foldl' (\acc i -> Just $ forwardStep lc i acc) Nothing inputs

-- | scanl' for lists of tensors unsing an `LSTMCell`
--
forwardScan
    :: (KnownDType d, KnownNat i, KnownNat h, KnownNat b)
    => LSTMCell d i h
    -> [Tensor d '[b, i]]
    -> [(Tensor d '[b, h], Tensor d '[b, h])]
forwardScan lc inputs =
    catMaybes $ scanl' (\acc i -> Just $ forwardStep lc i acc) Nothing inputs
