module Torch.Indef.Static.NN.Padding where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.NN as Dynamic

_spatialReflectionPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialReflectionPadding_updateOutput t0 t1 = Dynamic._spatialReflectionPadding_updateOutput (asDynamic t0) (asDynamic t1)
_spatialReflectionPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialReflectionPadding_updateGradInput t0 t1 t2 = Dynamic._spatialReflectionPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_spatialReplicationPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialReplicationPadding_updateOutput t0 t1 = Dynamic._spatialReplicationPadding_updateOutput (asDynamic t0) (asDynamic t1)
_spatialReplicationPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialReplicationPadding_updateGradInput t0 t1 t2 = Dynamic._spatialReplicationPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_volumetricReplicationPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricReplicationPadding_updateOutput t0 t1 = Dynamic._volumetricReplicationPadding_updateOutput (asDynamic t0) (asDynamic t1)
_volumetricReplicationPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricReplicationPadding_updateGradInput t0 t1 t2 = Dynamic._volumetricReplicationPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_temporalReflectionPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> IO ()
_temporalReflectionPadding_updateOutput t0 t1 = Dynamic._temporalReflectionPadding_updateOutput (asDynamic t0) (asDynamic t1)
_temporalReflectionPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> IO ()
_temporalReflectionPadding_updateGradInput t0 t1 t2 = Dynamic._temporalReflectionPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_temporalReplicationPadding_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> IO ()
_temporalReplicationPadding_updateOutput t0 t1 = Dynamic._temporalReplicationPadding_updateOutput (asDynamic t0) (asDynamic t1)
_temporalReplicationPadding_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> IO ()
_temporalReplicationPadding_updateGradInput t0 t1 t2 = Dynamic._temporalReplicationPadding_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)


