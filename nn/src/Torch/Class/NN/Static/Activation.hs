module Torch.Class.NN.Static.Activation where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions
import Control.Monad

class IsTensor t => Activation (t :: [Nat] -> *) where
  pReLU_updateOutput      :: t d -> t d -> t d -> IO ()
  pReLU_updateGradInput   :: t d -> t d -> t d -> t d -> IO ()
  pReLU_accGradParameters :: t d -> t d -> t d -> t d -> t d -> Double -> IO ()

  rReLU_updateOutput      :: t d -> t d -> t d -> Double -> Double -> Bool -> Bool -> Generator (t d) -> IO ()
  rReLU_updateGradInput   :: t d -> t d -> t d -> t d -> Double -> Double -> Bool -> Bool -> IO ()

  eLU_updateOutput        :: t d -> t d -> Double -> Double -> Bool -> IO ()
  eLU_updateGradInput     :: t d -> t d' -> t d'' -> Double -> Double -> IO ()

  leakyReLU_updateOutput       :: t d -> t d -> Double -> Bool -> IO ()
  leakyReLU_updateGradInput    :: t d -> t d -> t d -> Double -> Bool -> IO ()

  threshold_updateOutput    :: t d -> t d -> Double -> Double -> Bool -> IO ()
  threshold_updateGradInput :: t d -> t d -> t d -> Double -> Double -> Bool -> IO ()
