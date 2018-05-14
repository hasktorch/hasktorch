module Torch.Indef.Static.NN.Pooling where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.NN.Pooling as Dynamic


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

_temporalMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_temporalMaxPooling_updateOutput t0 t1 ix0 = Dynamic._temporalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_temporalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_temporalMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._temporalMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_spatialAdaptiveMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialAdaptiveMaxPooling_updateOutput t0 t1 ix0 = Dynamic._spatialAdaptiveMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_spatialAdaptiveMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> IO ()
_spatialAdaptiveMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._spatialAdaptiveMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_spatialFractionalMaxPooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IndexTensor d -> Tensor d -> IO ()
_spatialFractionalMaxPooling_updateOutput t0 t1 a0 a1 a2 a3 ix0 t2 = Dynamic._spatialFractionalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) a0 a1 a2 a3 (longAsDynamic ix0) (asDynamic t2)

_spatialFractionalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IndexTensor d -> IO ()
_spatialFractionalMaxPooling_updateGradInput t0 t1 t2 a0 a1 a2 a3 ix0 = Dynamic._spatialFractionalMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) a0 a1 a2 a3 (longAsDynamic ix0)

_spatialMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_spatialMaxPooling_updateOutput t0 t1 ix0 = Dynamic._spatialMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_spatialMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_spatialMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._spatialMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_spatialDilatedMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_spatialDilatedMaxPooling_updateOutput t0 t1 ix0 = Dynamic._spatialDilatedMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_spatialDilatedMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_spatialDilatedMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._spatialDilatedMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_spatialMaxUnpooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialMaxUnpooling_updateOutput t0 t1 ix0 = Dynamic._spatialMaxUnpooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_spatialMaxUnpooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> IO ()
_spatialMaxUnpooling_updateGradInput t0 t1 t2 ix0 = Dynamic._spatialMaxUnpooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_volumetricFractionalMaxPooling_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IndexTensor d -> Tensor d -> IO ()
_volumetricFractionalMaxPooling_updateOutput t0 t1 i0 i1 i2 i3 i4 i5 ix0 t2 = Dynamic._volumetricFractionalMaxPooling_updateOutput (asDynamic t0) (asDynamic t1) i0 i1 i2 i3 i4 i5 (longAsDynamic ix0) (asDynamic t2)

_volumetricFractionalMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IndexTensor d -> IO ()
_volumetricFractionalMaxPooling_updateGradInput t0 t1 t2 i0 i1 i2 i3 i4 i5 ix0 = Dynamic._volumetricFractionalMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) i0 i1 i2 i3 i4 i5 (longAsDynamic ix0)

_volumetricMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_volumetricMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_volumetricDilatedMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricDilatedMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricDilatedMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_volumetricDilatedMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> IO ()
_volumetricDilatedMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricDilatedMaxPooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_volumetricMaxUnpooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricMaxUnpooling_updateOutput t0 t1 ix0 = Dynamic._volumetricMaxUnpooling_updateOutput (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_volumetricMaxUnpooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricMaxUnpooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricMaxUnpooling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

_volumetricAdaptiveMaxPooling_updateOutput :: Tensor d -> Tensor d -> IndexTensor d -> Int -> Int -> Int -> IO ()
_volumetricAdaptiveMaxPooling_updateOutput t0 t1 ix0 = Dynamic._volumetricAdaptiveMaxPooling_updateOutput  (asDynamic t0) (asDynamic t1) (longAsDynamic ix0)

_volumetricAdaptiveMaxPooling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> IndexTensor d -> IO ()
_volumetricAdaptiveMaxPooling_updateGradInput t0 t1 t2 ix0 = Dynamic._volumetricAdaptiveMaxPooling_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (longAsDynamic ix0)

