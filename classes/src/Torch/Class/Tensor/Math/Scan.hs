module Torch.Class.Tensor.Math.Scan where

import Torch.Class.Types
import Data.Int

class TensorMathScan t where
  cumsum_  :: t -> t -> Int -> IO ()
  cumprod_ :: t -> t -> Int -> IO ()
