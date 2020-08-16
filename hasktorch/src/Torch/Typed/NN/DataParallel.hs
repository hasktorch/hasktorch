{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.NN.DataParallel where

import Control.Concurrent.Async
import Data.Kind
import GHC.TypeLits
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import Torch.NN (HasForward (..))
import qualified Torch.Tensor as D
import Torch.Typed.Autograd
import Torch.Typed.Device
import Torch.Typed.Optim

data ForwardConcurrentlyF = ForwardConcurrentlyF | ForwardConcurrentlyStochF

instance
  ( HasForward model input output
  ) =>
  Apply' ForwardConcurrentlyF (model, input) (Concurrently output)
  where
  apply' ForwardConcurrentlyF (model, input) = Concurrently . pure . forward model $ input
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
forwardConcurrently',
  forwardConcurrentlyStoch' ::
    forall devices' device' device model input output models inputs outputs.
    ( 'Just device ~ GetDevice model,
      'Just device ~ GetDevice input,
      HasScatter devices' device input inputs,
      HasReplicate devices' device model models,
      HZipWithM Concurrently ForwardConcurrentlyF models inputs outputs,
      HasGather device' devices' outputs output
    ) =>
    model ->
    input ->
    IO output
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

forwardConcurrently,
  forwardConcurrentlyStoch ::
    forall models inputs outputs.
    HZipWithM Concurrently ForwardConcurrentlyF models inputs outputs =>
    HList models ->
    HList inputs ->
    Concurrently (HList outputs)
forwardConcurrently = hzipWithM ForwardConcurrentlyF
forwardConcurrentlyStoch = hzipWithM ForwardConcurrentlyStochF

class HasGradConcurrently device' devices parameters losses gradients | device' devices parameters losses -> gradients where
  gradConcurrently :: HList parameters -> HList losses -> Concurrently (HList gradients)

data GradConcurrentlyF = GradConcurrentlyF

instance
  ( HasGrad (HList parameters) (HList gradients),
    ATen.Castable (HList gradients) [D.ATenTensor]
  ) =>
  Apply' GradConcurrentlyF (HList parameters, Loss device dtype) (Concurrently (HList gradients))
  where
  apply' GradConcurrentlyF (parameters, loss) = Concurrently . pure . grad loss $ parameters

instance
  ( HZipWithM Concurrently GradConcurrentlyF parameters losses gradients',
    ReduceGradients device' devices gradients' gradients
  ) =>
  HasGradConcurrently device' devices parameters losses gradients
  where
  gradConcurrently parameters losses =
    let gradients = hzipWithM GradConcurrentlyF parameters losses
     in reduceGradients @device' @devices <$> gradients

class ReduceGradients (device' :: (D.DeviceType, Nat)) (devices :: [(D.DeviceType, Nat)]) xxs ys | device' devices xxs -> ys where
  reduceGradients :: HList xxs -> HList ys

instance
  {-# OVERLAPS #-}
  ( HasToDevice device' device (HList xs) (HList ys)
  ) =>
  ReduceGradients device' (device ': '[]) ((HList (xs :: [Type])) ': '[]) ys
  where
  reduceGradients (xs :. HNil) = Torch.Typed.Device.toDevice @device' @device xs

data SumF = SumF

instance Num y => Apply' SumF (y, y) y where
  apply' _ = sum

instance
  {-# OVERLAPPABLE #-}
  ( HasToDevice device' device (HList xs) (HList ys),
    ReduceGradients device' devices xxs ys,
    HZipWith SumF ys ys ys,
    1 <= ListLength xxs
  ) =>
  ReduceGradients device' (device ': devices) ((HList (xs :: [Type])) ': xxs) ys
  where
  reduceGradients (xs :. xxs) = hzipWith SumF (Torch.Typed.Device.toDevice @device' @device xs) (reduceGradients @device' @devices @xxs xxs)
