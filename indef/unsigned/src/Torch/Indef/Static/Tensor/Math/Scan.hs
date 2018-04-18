module Torch.Indef.Static.Tensor.Math.Scan where

import Torch.Class.Tensor.Math.Scan
import Torch.Dimensions

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Scan ()

instance TensorMathScan (Tensor d) where
  _cumsum :: Tensor d -> Tensor d -> DimVal -> IO ()
  _cumsum r t = _cumsum (asDynamic r) (asDynamic t)

  _cumprod :: Tensor d -> Tensor d -> DimVal -> IO ()
  _cumprod r t = _cumprod (asDynamic r) (asDynamic t)

