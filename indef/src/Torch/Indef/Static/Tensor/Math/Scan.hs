module Torch.Indef.Static.Tensor.Math.Scan where

import Torch.Dimensions

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Scan as Dynamic

_cumsum :: Tensor d -> Tensor d -> DimVal -> IO ()
_cumsum r t = Dynamic._cumsum (asDynamic r) (asDynamic t)

_cumprod :: Tensor d -> Tensor d -> DimVal -> IO ()
_cumprod r t = Dynamic._cumprod (asDynamic r) (asDynamic t)

