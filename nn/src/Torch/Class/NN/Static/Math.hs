module Torch.Class.NN.Static.Math where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions

class IsTensor t => Math (t :: [Nat] -> *) where
  _abs_updateOutput
    :: t d   -- ^ input
    -> t d   -- ^ output
    -> IO ()

  _abs_updateGradInput
    :: (Product d ~ Product d')
    => t d   -- ^ input
    -> t d'  -- ^ gradOutput
    -> t d   -- ^ gradInput
    -> IO ()

  _sqrt_updateOutput    :: t d -> t d -> Double -> IO ()
  _sqrt_updateGradInput :: t d -> t d -> t d -> t d -> IO ()

  _square_updateOutput    :: t d -> t d -> IO ()
  _square_updateGradInput :: t d -> t d -> t d -> IO ()

  _logSigmoid_updateOutput      :: t d -> t d -> t d -> IO ()
  _logSigmoid_updateGradInput   :: t d -> t d -> t d -> t d -> IO ()

  _logSoftMax_updateOutput      :: t d -> t d -> DimReal (t d) -> IO ()
  _logSoftMax_updateGradInput   :: t d -> t d -> t d -> t d -> DimReal (t d) -> IO ()

  _sigmoid_updateOutput    :: t d -> t d -> IO ()
  _sigmoid_updateGradInput :: t d -> t d -> t d -> IO ()

  _softMax_updateOutput    :: t d -> t d -> DimReal (t d) -> IO ()
  _softMax_updateGradInput :: t d -> t d -> t d -> t d -> DimReal (t d) -> IO ()

  _softPlus_updateOutput    :: t d -> t d -> Double -> Double -> IO ()
  _softPlus_updateGradInput :: t d -> t d -> t d -> t d -> Double -> Double -> IO ()

  _softShrink_updateOutput    :: t d -> t d -> Double -> IO ()
  _softShrink_updateGradInput :: t d -> t d -> t d -> Double -> IO ()

  _tanh_updateOutput    :: t d -> t d -> IO ()
  _tanh_updateGradInput :: t d -> t d -> t d -> IO ()

  _hardTanh_updateOutput    :: t d -> t d -> Double -> Double -> Bool -> IO ()
  _hardTanh_updateGradInput :: t d -> t d -> t d -> Double -> Double -> Bool -> IO ()

abs_updateOutput :: Math t => t d -> IO (t d)
abs_updateOutput i = empty >>= \o -> _abs_updateOutput i o >> pure o

abs_updateGradInput :: (Product d ~ Product d', Math t) => t d -> t d' -> IO (t d)
abs_updateGradInput i gout = empty >>= \gin -> _abs_updateGradInput i gout gin >> pure gin



