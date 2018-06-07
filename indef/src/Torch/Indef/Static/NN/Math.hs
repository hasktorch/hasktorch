-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Math
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE TypeFamilies #-}
module Torch.Indef.Static.NN.Math where

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Data.Singletons.Prelude.List
import qualified Torch.Indef.Dynamic.NN as Dynamic

-- | abs forward pass (updates the output tensor)
abs_updateOutput :: Tensor d -> IO (Tensor d)
abs_updateOutput i = empty >>= \o -> Dynamic._abs_updateOutput (asDynamic i) (asDynamic o) >> pure o

-- | abs backward-update (updates the layer and bias tensors)
abs_updateGradInput
  :: (Product d ~ Product d')
  => Tensor d        -- ^ input
  -> Tensor d'       -- ^ gradOutput
  -> IO (Tensor d)   -- ^ gradInput
abs_updateGradInput i go =
  empty >>= \gi -> Dynamic._abs_updateGradInput (asDynamic i) (asDynamic go) (asDynamic gi) >> pure gi

-- |  sqrt forward pass (updates the output tensor)
_sqrt_updateOutput :: Tensor d -> Tensor d -> Double -> IO ()
_sqrt_updateOutput t0 t1 = Dynamic._sqrt_updateOutput (asDynamic t0) (asDynamic t1)
-- |  sqrt backward-update (updates the layer and bias tensors)
_sqrt_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_sqrt_updateGradInput t0 t1 t2 t3 = Dynamic._sqrt_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- |  square forward pass (updates the output tensor)
_square_updateOutput :: Tensor d -> Tensor d -> IO ()
_square_updateOutput t0 t1 = Dynamic._square_updateOutput (asDynamic t0) (asDynamic t1)
-- |  square backward-update (updates the layer and bias tensors)
_square_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_square_updateGradInput t0 t1 t2 = Dynamic._square_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  logSigmoid forward pass (updates the output tensor)
_logSigmoid_updateOutput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_logSigmoid_updateOutput t0 t1 t2 = Dynamic._logSigmoid_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- |  logSigmoid backward-update (updates the layer and bias tensors)
_logSigmoid_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_logSigmoid_updateGradInput t0 t1 t2 t3 = Dynamic._logSigmoid_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- |  logSoftMax forward pass (updates the output tensor)
_logSoftMax_updateOutput :: Tensor d -> Tensor d -> Integer -> IO ()
_logSoftMax_updateOutput t0 t1 = Dynamic._logSoftMax_updateOutput (asDynamic t0) (asDynamic t1)
-- |  logSoftMax backward-update (updates the layer and bias tensors)
_logSoftMax_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Integer -> IO ()
_logSoftMax_updateGradInput t0 t1 t2 t3 = Dynamic._logSoftMax_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- |  sigmoid forward pass (updates the output tensor)
_sigmoid_updateOutput :: Tensor d -> Tensor d -> IO ()
_sigmoid_updateOutput t0 t1 = Dynamic._sigmoid_updateOutput (asDynamic t0) (asDynamic t1)
-- |  sigmoid backward-update (updates the layer and bias tensors)
_sigmoid_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_sigmoid_updateGradInput t0 t1 t2 = Dynamic._sigmoid_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  softMax forward pass (updates the output tensor)
_softMax_updateOutput :: Tensor d -> Tensor d -> Integer -> IO ()
_softMax_updateOutput t0 t1 = Dynamic._softMax_updateOutput (asDynamic t0) (asDynamic t1)
-- |  softMax backward-update (updates the layer and bias tensors)
_softMax_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Integer -> IO ()
_softMax_updateGradInput t0 t1 t2 t3 = Dynamic._softMax_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- |  softPlus forward pass (updates the output tensor)
_softPlus_updateOutput :: Tensor d -> Tensor d -> Double -> Double -> IO ()
_softPlus_updateOutput t0 t1 = Dynamic._softPlus_updateOutput (asDynamic t0) (asDynamic t1)
-- |  softPlus backward-update (updates the layer and bias tensors)
_softPlus_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Double -> IO ()
_softPlus_updateGradInput t0 t1 t2 t3 = Dynamic._softPlus_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)

-- |  softShrink forward pass (updates the output tensor)
_softShrink_updateOutput :: Tensor d -> Tensor d -> Double -> IO ()
_softShrink_updateOutput t0 t1 = Dynamic._softShrink_updateOutput (asDynamic t0) (asDynamic t1)
-- |  softShrink backward-update (updates the layer and bias tensors)
_softShrink_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Double -> IO ()
_softShrink_updateGradInput t0 t1 t2 = Dynamic._softShrink_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  tanh forward pass (updates the output tensor)
_tanh_updateOutput :: Tensor d -> Tensor d -> IO ()
_tanh_updateOutput t0 t1 = Dynamic._tanh_updateOutput (asDynamic t0) (asDynamic t1)
-- |  tanh backward-update (updates the layer and bias tensors)
_tanh_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_tanh_updateGradInput t0 t1 t2 = Dynamic._tanh_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

-- |  hardTanh forward pass (updates the output tensor)
_hardTanh_updateOutput :: Tensor d -> Tensor d -> Double -> Double -> Bool -> IO ()
_hardTanh_updateOutput t0 t1 = Dynamic._hardTanh_updateOutput (asDynamic t0) (asDynamic t1)

-- |  hardTanh backward-update (updates the layer and bias tensors)
_hardTanh_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Double -> Double -> Bool -> IO ()
_hardTanh_updateGradInput t0 t1 t2 = Dynamic._hardTanh_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

