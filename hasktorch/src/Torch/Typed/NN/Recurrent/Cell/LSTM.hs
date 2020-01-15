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
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.NN                      as A
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Factories
import           Torch.Typed.Functional      hiding ( linear )
import           Torch.Typed.NN


-- | A specification for a long, short-term memory cell.
--
data LSTMCellSpec (inputDim :: Nat) (hiddenDim :: Nat)
                  (dtype :: D.DType)
                  (device :: (D.DeviceType, Nat))
  = LSTMCellSpec -- ^ Weights and biases are drawn from the standard normal distibution (having mean 0 and variance 1)
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)

-- | A long, short-term memory cell.
--
data LSTMCell (inputDim :: Nat) (hiddenDim :: Nat)
              (dtype :: D.DType)
              (device :: (D.DeviceType, Nat))
  =  LSTMCell { lstmCell_w_ih :: Parameter device dtype '[4 * hiddenDim, inputDim] -- ^ input-to-hidden weights
              , lstmCell_w_hh :: Parameter device dtype '[4 * hiddenDim, hiddenDim] -- ^ hidden-to-hidden weights
              , lstmCell_b_ih :: Parameter device dtype '[4 * hiddenDim] -- ^ input-to-hidden bias
              , lstmCell_b_hh :: Parameter device dtype '[4 * hiddenDim] -- ^ hidden-to-hidden bias
              }
  deriving Generic

instance ( KnownDevice device
         , KnownDType dtype
         , KnownNat inputDim
         , KnownNat hiddenDim
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable
       (LSTMCellSpec inputDim hiddenDim dtype device)
       (LSTMCell     inputDim hiddenDim dtype device)
 where
  sample LSTMCellSpec =
    LSTMCell
      <$> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)

-- | A single recurrent step of an `LSTMCell`
--
forwardStep
  :: forall inputDim hiddenDim batchSize dtype device
   . ( KnownDType dtype
     , KnownNat inputDim
     , KnownNat hiddenDim
     , KnownNat batchSize
     )
  => LSTMCell inputDim hiddenDim dtype device -- ^ The cell
  -> ( Tensor device dtype '[batchSize, hiddenDim]
     , Tensor device dtype '[batchSize, hiddenDim]
     ) -- ^ The current (Hidden, Cell) state
  -> Tensor device dtype '[batchSize, inputDim] -- ^ The input
  -> ( Tensor device dtype '[batchSize, hiddenDim]
     , Tensor device dtype '[batchSize, hiddenDim]
     ) -- ^ The subsequent (Hidden, Cell) state
forwardStep LSTMCell {..} = lstmCell (toDependent lstmCell_w_ih)
                                     (toDependent lstmCell_w_hh)
                                     (toDependent lstmCell_b_ih)
                                     (toDependent lstmCell_b_hh)

-- | foldl' for lists of tensors unsing an `LSTMCell`
--
forward
  :: forall inputDim hiddenDim batchSize dtype device
   . ( KnownDType dtype
     , KnownNat inputDim
     , KnownNat hiddenDim
     , KnownNat batchSize
     )
  => LSTMCell inputDim hiddenDim dtype device
  -> ( Tensor device dtype '[batchSize, hiddenDim]
     , Tensor device dtype '[batchSize, hiddenDim]
     ) -- ^ The initial (Hidden, Cell) state
  -> [Tensor device dtype '[batchSize, inputDim]] -- ^ The list of inputs
  -> ( Tensor device dtype '[batchSize, hiddenDim]
     , Tensor device dtype '[batchSize, hiddenDim]
     ) -- ^ The final (Hidden, Cell) state
forward cell = foldl' (forwardStep cell)

-- | scanl' for lists of tensors unsing an `LSTMCell`
--
forwardScan
  :: forall inputDim hiddenDim batchSize dtype device
   . ( KnownDType dtype
     , KnownNat inputDim
     , KnownNat hiddenDim
     , KnownNat batchSize
     )
  => LSTMCell inputDim hiddenDim dtype device
  -> ( Tensor device dtype '[batchSize, hiddenDim]
     , Tensor device dtype '[batchSize, hiddenDim]
     ) -- ^ The initial (Hidden, Cell) state
  -> [Tensor device dtype '[batchSize, inputDim]] -- ^ The list of inputs
  -> [ ( Tensor device dtype '[batchSize, hiddenDim]
       , Tensor device dtype '[batchSize, hiddenDim]
       )
     ] -- ^ All subsequent (Hidden, Cell) states
forwardScan cell = scanl' (forwardStep cell)
