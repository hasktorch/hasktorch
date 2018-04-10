module Torch.Class.Tensor.Sort where

import Torch.Class.Types
import Torch.Dimensions

class TensorSort t where
  _sort :: (t, IndexDynamic t) -> t -> DimVal -> SortOrder -> IO ()

class GPUTensorSort t where
  sortKeyValueInplace :: t -> IndexDynamic t -> Int -> Int -> IO ()
