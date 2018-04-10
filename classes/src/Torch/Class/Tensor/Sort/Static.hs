module Torch.Class.Tensor.Sort.Static where

import Torch.Class.Types
import Torch.Dimensions

class TensorSort t where
  _sort :: (t d, IndexTensor (t d) d) -> t d -> DimVal -> SortOrder -> IO ()

-- class GPUTensorSort t where
--   sortKeyValueInplace :: t -> IndexDynamic t -> Int -> Int -> IO ()
