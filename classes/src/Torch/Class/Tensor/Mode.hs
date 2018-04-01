module Torch.Class.Tensor.Mode where

import Torch.Class.Types
import Torch.Dimensions


class TensorMode t where
  mode_ :: (t, IndexDynamic t) -> t -> DimVal -> Maybe KeepDim -> IO ()

