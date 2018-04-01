module Torch.Class.Tensor.Math.Scan where

import Torch.Class.Types
import Data.Int
import Torch.Dimensions

class TensorMathScan t where
  cumsum_  :: t -> t -> DimVal -> IO ()
  cumprod_ :: t -> t -> DimVal -> IO ()
