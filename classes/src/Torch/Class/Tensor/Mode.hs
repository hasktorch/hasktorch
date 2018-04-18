module Torch.Class.Tensor.Mode where

import Torch.Class.Types
import Torch.Dimensions


class TensorMode t where
  _mode :: (t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ()

