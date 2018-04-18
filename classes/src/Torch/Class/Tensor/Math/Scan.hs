module Torch.Class.Tensor.Math.Scan where

import Torch.Class.Types
import Data.Int
import Torch.Dimensions

class TensorMathScan t where
  _cumsum  :: t -> t -> DimVal -> IO ()
  _cumprod :: t -> t -> DimVal -> IO ()
