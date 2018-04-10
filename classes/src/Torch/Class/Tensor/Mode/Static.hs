module Torch.Class.Tensor.Mode.Static where

import Torch.Class.Types
import Torch.Dimensions

class TensorMode t where
  _mode :: (t d, IndexTensor (t d) d) -> t d -> DimVal -> Maybe KeepDim -> IO ()

