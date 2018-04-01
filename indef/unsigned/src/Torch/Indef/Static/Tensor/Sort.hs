module Torch.Indef.Static.Tensor.Sort where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Sort.Static as Class
import qualified Torch.Class.Tensor.Sort as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Sort ()

instance Class.TensorSort Tensor where
  sort_ :: (Tensor d', IndexTensor d') -> Tensor d -> DimVal -> SortOrder -> IO ()
  sort_ (r, ix) t = Dynamic.sort_ (asDynamic r, longAsDynamic ix) (asDynamic t)


