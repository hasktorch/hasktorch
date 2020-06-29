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

module Torch.Typed.NN.Recurrent.Cell.GRU where

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
import           Torch.Typed.NN.Dropout


-- | A specification for a gated recurrent unit (GRU) cell.
--
data GRUCellSpec (inputDim :: Nat) (hiddenDim :: Nat)
                  (dtype :: D.DType)
                  (device :: (D.DeviceType, Nat))
  = GRUCellSpec -- ^ Weights and biases are drawn from the standard normal distibution (having mean 0 and variance 1)
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)

-- | A gated recurrent unit (GRU) cell.
--
data GRUCell (inputDim :: Nat) (hiddenDim :: Nat)
              (dtype :: D.DType)
              (device :: (D.DeviceType, Nat))
  =  GRUCell { gruCell_w_ih :: Parameter device dtype '[3 * hiddenDim, inputDim] -- ^ input-to-hidden weights
              , gruCell_w_hh :: Parameter device dtype '[3 * hiddenDim, hiddenDim] -- ^ hidden-to-hidden weights
              , gruCell_b_ih :: Parameter device dtype '[3 * hiddenDim] -- ^ input-to-hidden bias
              , gruCell_b_hh :: Parameter device dtype '[3 * hiddenDim] -- ^ hidden-to-hidden bias
              }
  deriving Generic

instance ( KnownDevice device
         , KnownDType dtype
         , KnownNat inputDim
         , KnownNat hiddenDim
         , RandDTypeIsValid device dtype
         )
  => A.Randomizable
       (GRUCellSpec inputDim hiddenDim dtype device)
       (GRUCell     inputDim hiddenDim dtype device)
 where
  sample GRUCellSpec =
    GRUCell
      <$> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)

-- | A single recurrent step of a `GRUCell`
--
gruCellForward
  :: forall inputDim hiddenDim batchSize dtype device
   . ( KnownDType dtype
     , KnownNat inputDim
     , KnownNat hiddenDim
     , KnownNat batchSize
     )
  => GRUCell inputDim hiddenDim dtype device -- ^ The cell
  -> Tensor device dtype '[batchSize, hiddenDim]-- ^ The current Hidden state
  -> Tensor device dtype '[batchSize, inputDim]-- ^ The input
  -> Tensor device dtype '[batchSize, hiddenDim]-- ^ The subsequent Hidden state
gruCellForward GRUCell {..} = gruCell (toDependent gruCell_w_ih)
                                      (toDependent gruCell_w_hh)
                                      (toDependent gruCell_b_ih)
                                      (toDependent gruCell_b_hh)

-- | foldl' for lists of tensors unsing a `GRUCell`
--
gruFold
  :: forall inputDim hiddenDim batchSize dtype device
   . ( KnownDType dtype
     , KnownNat inputDim
     , KnownNat hiddenDim
     , KnownNat batchSize
     )
  => GRUCell inputDim hiddenDim dtype device
  -> Tensor device dtype '[batchSize, hiddenDim] -- ^ The initial Hidden state
  -> [Tensor device dtype '[batchSize, inputDim]] -- ^ The list of inputs
  -> Tensor device dtype '[batchSize, hiddenDim] -- ^ The final Hidden state
gruFold cell = foldl' (gruCellForward cell)

-- | scanl' for lists of tensors unsing a `GRUCell`
--
gruCellScan
  :: forall inputDim hiddenDim batchSize dtype device
   . ( KnownDType dtype
     , KnownNat inputDim
     , KnownNat hiddenDim
     , KnownNat batchSize
     )
  => GRUCell inputDim hiddenDim dtype device
  -> Tensor device dtype '[batchSize, hiddenDim] -- ^ The initial Hidden state
  -> [Tensor device dtype '[batchSize, inputDim]] -- ^ The list of inputs
  -> [Tensor device dtype '[batchSize, hiddenDim]] -- ^ All subsequent Hidden states
gruCellScan cell = scanl' (gruCellForward cell)
