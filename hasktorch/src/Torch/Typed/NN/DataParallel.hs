{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}

module Torch.Typed.NN.DataParallel (
    forwardConcurrently
) where

import           Torch.HList
import           Control.Concurrent.Async
import           GHC.TypeLits
import           System.IO.Unsafe

import           Torch.Typed.Aux
import           Torch.Typed.Device
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Functional
import           Torch.Typed.NN

data ForwardConcurrently = ForwardConcurrently

instance 
  ( HasForward model input (IO output)
  ) => Apply' ForwardConcurrently (model, input) (Concurrently output) where
  apply' _ (model, input) = Concurrently $ forward model input

-- Run a `model` concurrently on an `input`.
--
-- The `model` is replicated over the supplied `devices'`, and the `input` is scattered
-- over them as well. Then the `forward` function of the replicated `models` is run
-- concurrently on the scattered `inputs`. Finally, the `outputs` are gathered on the
-- target `device'`
--
-- >>> model <- A.sample (LinearSpec @1 @1 @'D.Float @'( 'D.CPU, 0))
-- >>> t = ones @'[2, 1] @'D.Float @'( 'D.CPU, 0)
--
-- >>> :t forward model t
-- forward model t :: IO (Tensor '( 'D.CPU, 0) 'D.Float '[2, 1])
-- >>> forward model t
-- Tensor Float [2,1] [[ 0.2478   ],
--                     [ 0.2478   ]]
--
-- >>> :t forwardConcurrently @'[ '( 'D.CPU, 0), '( 'D.CUDA, 0)] @'( 'D.CPU, 0) model t
-- forwardConcurrently @'[ '( 'D.CPU, 0), '( 'D.CUDA, 0)] @'( 'D.CPU, 0) model t
--   :: IO (Tensor '( 'D.CPU, 0) 'D.Float '[2, 1])
-- >>> forwardConcurrently @'[ '( 'D.CPU, 0), '( 'D.CUDA, 0)] @'( 'D.CPU, 0) model t
-- Tensor Float [2,1] [[ 0.2478   ],
--                     [ 0.2478   ]]
forwardConcurrently
  :: forall devices' device' device model input output models inputs outputs
   . ( 'Just device ~ GetDevice model
     , 'Just device ~ GetDevice input
     , HasScatter devices' device input inputs
     , HasReplicate devices' device model models
     , HZipWithM Concurrently ForwardConcurrently models inputs outputs
     , HasGather device' devices' outputs output
     )
  => model
  -> input
  -> IO output
forwardConcurrently model input = do
  let models = Torch.Typed.Device.replicate @devices' @device model
      inputs = scatter @devices' @device input
  outputs <- runConcurrently $ hzipWithM ForwardConcurrently models inputs
  let output = gather @device' @devices' outputs
  return output
