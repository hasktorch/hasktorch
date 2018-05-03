module Torch.Class.NN.Static.Layers where

import Torch.Class.Tensor.Static
import Torch.Dimensions
import Control.Monad

class FusedLayers (t :: [Nat] -> *) where
  _sparseLinear_updateOutput       :: t d -> t d -> t d -> t d -> IO ()
  _sparseLinear_accGradParameters  :: t d -> t d -> t d -> t d -> t d -> t d -> Double -> Double -> IO ()
  _sparseLinear_zeroGradParameters :: t d -> t d -> t d -> IO ()
  _sparseLinear_updateParameters   :: t d -> t d -> t d -> t d -> t d -> Double -> IO ()

  _gatedLinear_updateOutput     :: t d -> t d -> Int -> IO ()
  _gatedLinear_updateGradInput  :: t d -> t d -> t d -> Int -> IO ()

  _gRUFused_updateOutput        :: t d -> t d -> t d -> t d -> t d -> t d -> t d -> IO ()
  _gRUFused_updateGradInput     :: t d -> t d -> t d -> t d -> t d -> IO ()

  _lSTMFused_updateOutput       :: t d -> t d -> t d -> t d -> t d -> t d -> t d -> IO ()
  _lSTMFused_updateGradInput    :: t d -> t d -> t d -> t d -> t d -> t d -> t d -> IO ()


