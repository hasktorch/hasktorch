module Torch.Indef.Static.Tensor.Sort where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Sort.Static as Class
import qualified Torch.Class.Tensor.Sort as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Sort ()

instance Class.TensorSort Tensor where
  _sort :: (Tensor d', IndexTensor '[n]) -> Tensor d -> DimVal -> SortOrder -> IO ()
  _sort (r, ix) t = Dynamic._sort (asDynamic r, longAsDynamic ix) (asDynamic t)


