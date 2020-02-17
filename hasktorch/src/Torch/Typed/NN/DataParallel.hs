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

module Torch.Typed.NN.DataParallel where

import           Data.Kind
import           Control.Concurrent.Async
import           GHC.TypeLits
import           System.IO.Unsafe

import           Torch.HList
import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Tensor as D
import qualified Torch.Device as D
import qualified Torch.DType as D
import           Torch.Typed.Aux
import           Torch.Typed.Autograd
import           Torch.Typed.Device
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Functional
import           Torch.Typed.Factories
import           Torch.Typed.Optim
import           Torch.Typed.NN

data ForwardConcurrentlyF = ForwardConcurrentlyF | ForwardConcurrentlyStochF

instance 
  ( HasForward model input output
  ) => Apply' ForwardConcurrentlyF (model, input) (Concurrently output) where
  apply' ForwardConcurrentlyF      (model, input) = Concurrently . pure . forward model $ input
  apply' ForwardConcurrentlyStochF (model, input) = Concurrently . forwardStoch model $ input

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
-- >>> :t forwardConcurrently' @'[ '( 'D.CPU, 0), '( 'D.CUDA, 0)] @'( 'D.CPU, 0) model t
-- forwardConcurrently' @'[ '( 'D.CPU, 0), '( 'D.CUDA, 0)] @'( 'D.CPU, 0) model t
--   :: IO (Tensor '( 'D.CPU, 0) 'D.Float '[2, 1])
-- >>> forwardConcurrently' @'[ '( 'D.CPU, 0), '( 'D.CUDA, 0)] @'( 'D.CPU, 0) model t
-- Tensor Float [2,1] [[ 0.2478   ],
--                     [ 0.2478   ]]
forwardConcurrently', forwardConcurrentlyStoch'
  :: forall devices' device' device model input output models inputs outputs
   . ( 'Just device ~ GetDevice model
     , 'Just device ~ GetDevice input
     , HasScatter devices' device input inputs
     , HasReplicate devices' device model models
     , HZipWithM Concurrently ForwardConcurrentlyF models inputs outputs
     , HasGather device' devices' outputs output
     )
  => model
  -> input
  -> IO output
forwardConcurrently' model input = do
  let models = Torch.Typed.Device.replicate @devices' @device @model @models model
      inputs = scatter @devices' @device @input @inputs input
  outputs <- runConcurrently $ forwardConcurrently models inputs
  let output = gather @device' @devices' @outputs @output outputs
  return output
forwardConcurrentlyStoch' model input = do
  let models = Torch.Typed.Device.replicate @devices' @device @model @models model
      inputs = scatter @devices' @device @input @inputs input
  outputs <- runConcurrently $ forwardConcurrentlyStoch models inputs
  let output = gather @device' @devices' @outputs @output outputs
  return output

forwardConcurrently, forwardConcurrentlyStoch
  :: forall models inputs outputs
   . HZipWithM Concurrently ForwardConcurrentlyF models inputs outputs
  => HList models
  -> HList inputs
  -> Concurrently (HList outputs)
forwardConcurrently = hzipWithM ForwardConcurrentlyF
forwardConcurrentlyStoch = hzipWithM ForwardConcurrentlyStochF

class HasGradConcurrently device' devices parameters losses gradients | device' devices parameters losses -> gradients where
  gradConcurrently :: HList parameters -> HList losses -> Concurrently (HList gradients)

data GradConcurrentlyF = GradConcurrentlyF

instance 
  ( HasGrad (HList parameters) (HList gradients)
  , ATen.Castable (HList gradients) [D.ATenTensor]
  ) => Apply' GradConcurrentlyF (HList parameters, Loss device dtype) (Concurrently (HList gradients)) where
  apply' GradConcurrentlyF (parameters, loss) = Concurrently . pure  . grad loss $ parameters

instance
  ( HZipWithM Concurrently GradConcurrentlyF parameters losses ((HList (xs :: [Type])) ': xxs)
  , HReplicateFD (ListLength xs) (HList ('[] :: [Type])) acc
  , HFoldr HZipF (HList acc) ((HList xs) ': xxs) res
  , devices ~ GetDevices losses
  , res ~ HList res'
  , HasCoalesce device' devices res' gradients
  ) => HasGradConcurrently device' devices parameters losses gradients where
  gradConcurrently parameters losses = 
    let gradients = hzipWithM GradConcurrentlyF parameters losses
    in  (coalesce @device' @devices . htranspose) <$> gradients

class HasCoalesce (device' :: (D.DeviceType, Nat)) (devices :: [(D.DeviceType, Nat)]) xxs ys | device' devices xxs -> ys where
  coalesce :: HList xxs -> HList ys

instance HasCoalesce device' devices ('[] :: [Type]) ('[] :: [Type]) where
  coalesce = const HNil

instance
  ( HasCoalesce device' devices xxs ys
  , HasReduceAdd device' devices xs y
  ) => HasCoalesce device' devices ((HList xs) ': xxs) (y ': ys) where
  coalesce (xs :. xxs) = reduceAdd @device' @devices xs :. coalesce @device' @devices xxs

class HasReduceAdd (device' :: (D.DeviceType, Nat)) (devices :: [(D.DeviceType, Nat)]) fs g | device' devices fs -> g where
  reduceAdd :: HList fs -> g

instance  {-# OVERLAPS #-}
  (HasToDevice device' device f g) => HasReduceAdd device' (device ': '[]) (f ': '[]) g where
  reduceAdd (f :. HNil) = Torch.Typed.Device.toDevice @device' @device f

instance  {-# OVERLAPPABLE #-}
  ( Num g
  , 1 <= ListLength devices
  , HasToDevice device' device f g
  , HasReduceAdd device' devices fs g
  ) => HasReduceAdd device' (device ': devices) (f ': fs) g where
  reduceAdd (f :. fs) = Torch.Typed.Device.toDevice @device' @device f + reduceAdd @device' @devices fs

testReduceAdd = reduceAdd @'( 'D.CPU, 0) @'[ '( 'D.CUDA, 0), '( 'D.CUDA, 1)] (ones @'[1] @'D.Float @'( 'D.CUDA, 0) :. ones @'[1] @'D.Float @'( 'D.CUDA, 1) :. HNil)
