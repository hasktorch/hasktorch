module Torch.Indef.Static.NN.Sampling where

import Control.Monad
import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.NN as Dynamic

-- 1d sampling
_temporalUpSamplingNearest_updateOutput :: Tensor d -> Tensor d -> Int -> IO ()
_temporalUpSamplingNearest_updateOutput t0 t1 = Dynamic._temporalUpSamplingNearest_updateOutput (asDynamic t0) (asDynamic t1)
_temporalUpSamplingNearest_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_temporalUpSamplingNearest_updateGradInput t0 t1 t2 = Dynamic._temporalUpSamplingNearest_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_temporalUpSamplingLinear_updateOutput :: Tensor d -> Tensor d -> Int -> IO ()
_temporalUpSamplingLinear_updateOutput t0 t1 = Dynamic._temporalUpSamplingLinear_updateOutput (asDynamic t0) (asDynamic t1)
_temporalUpSamplingLinear_updateGradInput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_temporalUpSamplingLinear_updateGradInput t0 t1 = Dynamic._temporalUpSamplingLinear_updateGradInput (asDynamic t0) (asDynamic t1)

-- 2d sampling
_spatialSubSampling_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialSubSampling_updateOutput t0 t1 t2 t3 = Dynamic._spatialSubSampling_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)
_spatialSubSampling_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> IO ()
_spatialSubSampling_updateGradInput t0 t1 t2 t3 = Dynamic._spatialSubSampling_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)
_spatialSubSampling_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Double -> IO ()
_spatialSubSampling_accGradParameters t0 t1 t2 t3 = Dynamic._spatialSubSampling_accGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)
_spatialUpSamplingNearest_updateOutput :: Tensor d -> Tensor d -> Int -> IO ()
_spatialUpSamplingNearest_updateOutput t0 t1 = Dynamic._spatialUpSamplingNearest_updateOutput (asDynamic t0) (asDynamic t1)
_spatialUpSamplingNearest_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_spatialUpSamplingNearest_updateGradInput t0 t1 t2 = Dynamic._spatialUpSamplingNearest_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_spatialUpSamplingBilinear_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> IO ()
_spatialUpSamplingBilinear_updateOutput t0 t1 = Dynamic._spatialUpSamplingBilinear_updateOutput (asDynamic t0) (asDynamic t1)
_spatialUpSamplingBilinear_updateGradInput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_spatialUpSamplingBilinear_updateGradInput t0 t1 = Dynamic._spatialUpSamplingBilinear_updateGradInput (asDynamic t0) (asDynamic t1)
_spatialGridSamplerBilinear_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_spatialGridSamplerBilinear_updateOutput t0 t1 t2 = Dynamic._spatialGridSamplerBilinear_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_spatialGridSamplerBilinear_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_spatialGridSamplerBilinear_updateGradInput t0 t1 t2 t3 t4 = Dynamic._spatialGridSamplerBilinear_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)

-- 3d sampling
_volumetricGridSamplerBilinear_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_volumetricGridSamplerBilinear_updateOutput t0 t1 t2 = Dynamic._volumetricGridSamplerBilinear_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_volumetricGridSamplerBilinear_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_volumetricGridSamplerBilinear_updateGradInput t0 t1 t2 t3 t4 = Dynamic._volumetricGridSamplerBilinear_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)
_volumetricUpSamplingNearest_updateOutput :: Tensor d -> Tensor d -> Int -> IO ()
_volumetricUpSamplingNearest_updateOutput t0 t1 = Dynamic._volumetricUpSamplingNearest_updateOutput (asDynamic t0) (asDynamic t1)
_volumetricUpSamplingNearest_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_volumetricUpSamplingNearest_updateGradInput t0 t1 t2 = Dynamic._volumetricUpSamplingNearest_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)
_volumetricUpSamplingTrilinear_updateOutput :: Tensor d -> Tensor d -> Int -> Int -> Int -> IO ()
_volumetricUpSamplingTrilinear_updateOutput t0 t1 = Dynamic._volumetricUpSamplingTrilinear_updateOutput (asDynamic t0) (asDynamic t1)
_volumetricUpSamplingTrilinear_updateGradInput :: Tensor d -> Tensor d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_volumetricUpSamplingTrilinear_updateGradInput t0 t1 = Dynamic._volumetricUpSamplingTrilinear_updateGradInput (asDynamic t0) (asDynamic t1)
