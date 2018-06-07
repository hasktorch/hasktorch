-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Padding
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.NN.Padding where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.NN as Dynamic

-- |  spatialReflectionPadding forward pass (updates the output tensor)
_spatialReflectionPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialReflectionPadding_updateOutput t0 t1 = Dynamic._spatialReflectionPadding_updateOutput (asDynamic t0) (asDynamic t1)
-- |  spatialReflectionPadding backward-update (updates the layer and bias tensors)
_spatialReflectionPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialReflectionPadding_updateGradInput t0 t1 t2 = Dynamic._spatialReflectionPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- |  spatialReplicationPadding forward pass (updates the output tensor)
_spatialReplicationPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialReplicationPadding_updateOutput t0 t1 = Dynamic._spatialReplicationPadding_updateOutput (asDynamic t0) (asDynamic t1)
-- |  spatialReplicationPadding backward-update (updates the layer and bias tensors)
_spatialReplicationPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialReplicationPadding_updateGradInput t0 t1 t2 = Dynamic._spatialReplicationPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- |  volumetricReplicationPadding forward pass (updates the output tensor)
_volumetricReplicationPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricReplicationPadding_updateOutput t0 t1 = Dynamic._volumetricReplicationPadding_updateOutput (asDynamic t0) (asDynamic t1)
-- |  volumetricReplicationPadding backward-update (updates the layer and bias tensors)
_volumetricReplicationPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricReplicationPadding_updateGradInput t0 t1 t2 = Dynamic._volumetricReplicationPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- |  temporalReflectionPadding forward pass (updates the output tensor)
_temporalReflectionPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> IO ()
_temporalReflectionPadding_updateOutput t0 t1 = Dynamic._temporalReflectionPadding_updateOutput (asDynamic t0) (asDynamic t1)
-- |  temporalReflectionPadding backward-update (updates the layer and bias tensors)
_temporalReflectionPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> IO ()
_temporalReflectionPadding_updateGradInput t0 t1 t2 = Dynamic._temporalReflectionPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
-- |  temporalReplicationPadding forward pass (updates the output tensor)
_temporalReplicationPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> IO ()
_temporalReplicationPadding_updateOutput t0 t1 = Dynamic._temporalReplicationPadding_updateOutput (asDynamic t0) (asDynamic t1)
-- |  temporalReplicationPadding backward-update (updates the layer and bias tensors)
_temporalReplicationPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> IO ()
_temporalReplicationPadding_updateGradInput t0 t1 t2 = Dynamic._temporalReplicationPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)


