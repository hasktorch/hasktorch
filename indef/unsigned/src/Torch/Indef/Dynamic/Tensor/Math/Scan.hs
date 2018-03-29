module Torch.Indef.Dynamic.Tensor.Math.Scan where

import Torch.Class.Tensor.Math.Scan
import Torch.Indef.Types

import qualified Torch.Sig.Tensor.Math.Scan as Sig

instance TensorMathScan Dynamic where
  cumsum_ :: Dynamic -> Dynamic -> Int -> IO ()
  cumsum_ t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_cumsum (fromIntegral i0)

  cumprod_ :: Dynamic -> Dynamic -> Int -> IO ()
  cumprod_ t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_cumprod (fromIntegral i0)

