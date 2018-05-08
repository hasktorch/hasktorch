module Torch.Indef.Static.NN.Pooling where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.NN as Dynamic


_featureLPPooling_updateOutput :: Tensor d -> Tensor d -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateOutput t0 t1 = Dynamic._featureLPPooling_updateOutput (asDynamic t0) (asDynamic t1)
_featureLPPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Int -> Int -> Bool -> IO ()
_featureLPPooling_updateGradInput t0 t1 t2 t3 = Dynamic._featureLPPooling_updateGradInput (asDynamic t0) (asDynamic t1)
 (asDynamic t2) (asDynamic t3)
_spatialAdaptiveAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> IO ()
_spatialAdaptiveAveragePooling_updateOutput t0 t1 = Dynamic._spatialAdaptiveAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)
_spatialAdaptiveAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_spatialAdaptiveAveragePooling_updateGradInput t0 t1 t2 = Dynamic._spatialAdaptiveAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

_spatialAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateOutput t0 t1 = Dynamic._spatialAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)
_spatialAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_spatialAveragePooling_updateGradInput t0 t1 t2 = Dynamic._spatialAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

_volumetricAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateOutput t0 t1 =
  Dynamic._volumetricAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)
_volumetricAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
_volumetricAveragePooling_updateGradInput t0 t1 t2 =
  Dynamic._volumetricAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

_volumetricAdaptiveAveragePooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveAveragePooling_updateOutput t0 t1 = Dynamic._volumetricAdaptiveAveragePooling_updateOutput (asDynamic t0) (asDynamic t1)
_volumetricAdaptiveAveragePooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_volumetricAdaptiveAveragePooling_updateGradInput t0 t1 t2 =
  Dynamic._volumetricAdaptiveAveragePooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
