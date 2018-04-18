module Torch.Indef.Dynamic.Tensor.Math.Scan where

import Torch.Class.Tensor.Math.Scan
import Torch.Indef.Types
import Torch.Dimensions


import qualified Torch.Sig.Tensor.Math.Scan as Sig

instance TensorMathScan Dynamic where
  _cumsum :: Dynamic -> Dynamic -> DimVal -> IO ()
  _cumsum t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_cumsum (fromIntegral i0)

  _cumprod :: Dynamic -> Dynamic -> DimVal -> IO ()
  _cumprod t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_cumprod (fromIntegral i0)

