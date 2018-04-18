module Torch.Class.Tensor.Mode.Static where

import Torch.Class.Types
import Torch.Dimensions

class TensorMode t where
  _mode :: (t d, IndexTensor t '[n]) -> t d -> DimVal -> Maybe KeepDim -> IO ()

