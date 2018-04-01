module Torch.Indef.Static.Tensor.Math.Scan where

import Torch.Class.Tensor.Math.Scan
import Torch.Dimensions

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Scan ()

instance TensorMathScan (Tensor d) where
  cumsum_ :: Tensor d -> Tensor d -> DimVal -> IO ()
  cumsum_ r t = cumsum_ (asDynamic r) (asDynamic t)

  cumprod_ :: Tensor d -> Tensor d -> DimVal -> IO ()
  cumprod_ r t = cumprod_ (asDynamic r) (asDynamic t)

