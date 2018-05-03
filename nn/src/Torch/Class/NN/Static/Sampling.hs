module Torch.Class.NN.Static.Sampling where

import Torch.Class.Tensor.Static
import Torch.Dimensions
import Control.Monad

class IsTensor t => TemporalSampling (t :: [Nat] -> *) where
  temporalUpSamplingNearest_updateOutput     :: t d -> t d -> Int -> IO ()
  temporalUpSamplingNearest_updateGradInput  :: t d -> t d -> t d -> Int -> IO ()
  temporalUpSamplingLinear_updateOutput      :: t d -> t d -> Int -> IO ()
  temporalUpSamplingLinear_updateGradInput   :: t d -> t d -> Int -> Int -> Int -> Int -> IO ()

class IsTensor t => SpatialSampling (t :: [Nat] -> *) where
  spatialSubSampling_updateOutput            :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialSubSampling_updateGradInput         :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialSubSampling_accGradParameters       :: t d -> t d -> t d -> t d -> Int -> Int -> Int -> Int -> Double -> IO ()
  spatialUpSamplingNearest_updateOutput      :: t d -> t d -> Int -> IO ()
  spatialUpSamplingNearest_updateGradInput   :: t d -> t d -> t d -> Int -> IO ()
  spatialUpSamplingBilinear_updateOutput     :: t d -> t d -> Int -> Int -> IO ()
  spatialUpSamplingBilinear_updateGradInput  :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  spatialGridSamplerBilinear_updateOutput    :: t d -> t d -> t d -> Int -> IO ()
  spatialGridSamplerBilinear_updateGradInput :: t d -> t d -> t d -> t d -> t d -> Int -> IO ()

class IsTensor t => VolumetricSampling (t :: [Nat] -> *) where
  volumetricGridSamplerBilinear_updateOutput    :: t d -> t d -> t d -> Int -> IO ()
  volumetricGridSamplerBilinear_updateGradInput :: t d -> t d -> t d -> t d -> t d -> Int -> IO ()
  volumetricUpSamplingNearest_updateOutput      :: t d -> t d -> Int -> IO ()
  volumetricUpSamplingNearest_updateGradInput   :: t d -> t d -> t d -> Int -> IO ()
  volumetricUpSamplingTrilinear_updateOutput    :: t d -> t d -> Int -> Int -> Int -> IO ()
  volumetricUpSamplingTrilinear_updateGradInput :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
