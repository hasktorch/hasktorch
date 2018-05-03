module Torch.Class.NN.Static.Pooling where

import Torch.Class.Tensor.Static
import Torch.Dimensions
import Control.Monad

class IsTensor t => Pooling (t :: [Nat] -> *) where
  featureLPPooling_updateOutput    :: t d -> t d -> Double -> Int -> Int -> Bool -> IO ()
  featureLPPooling_updateGradInput :: t d -> t d -> t d -> t d -> Double -> Int -> Int -> Bool -> IO ()

  spatialAdaptiveAveragePooling_updateOutput    :: t d -> t d -> Int -> Int -> IO ()
  spatialAdaptiveAveragePooling_updateGradInput :: t d -> t d -> t d -> IO ()

  spatialAveragePooling_updateOutput            :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
  spatialAveragePooling_updateGradInput         :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()

  volumetricAveragePooling_updateOutput    :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()
  volumetricAveragePooling_updateGradInput :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Bool -> Bool -> IO ()

  volumetricAdaptiveAveragePooling_updateOutput    :: t d -> t d -> Int -> Int -> Int -> IO ()
  volumetricAdaptiveAveragePooling_updateGradInput :: t d -> t d -> t d -> IO ()
