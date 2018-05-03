module Torch.Class.NN.Static.Padding where

import Torch.Class.Tensor.Static
import Torch.Dimensions
import Control.Monad

class IsTensor t => Padding (t :: [Nat] -> *) where
  spatialReflectionPadding_updateOutput    :: t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialReflectionPadding_updateGradInput :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialReplicationPadding_updateOutput    :: t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  spatialReplicationPadding_updateGradInput :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> IO ()
  volumetricReplicationPadding_updateOutput    :: t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  volumetricReplicationPadding_updateGradInput :: t d -> t d -> t d -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
  temporalReflectionPadding_updateOutput    :: t d -> t d -> Int -> Int -> IO ()
  temporalReflectionPadding_updateGradInput :: t d -> t d -> t d -> Int -> Int -> IO ()
  temporalReplicationPadding_updateOutput    :: t d -> t d -> Int -> Int -> IO ()
  temporalReplicationPadding_updateGradInput :: t d -> t d -> t d -> Int -> Int -> IO ()


