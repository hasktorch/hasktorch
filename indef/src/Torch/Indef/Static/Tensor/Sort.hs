module Torch.Indef.Static.Tensor.Sort where

import Torch.Dimensions

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Sort as Dynamic

_sort :: (Tensor d', IndexTensor '[n]) -> Tensor d -> DimVal -> SortOrder -> IO ()
_sort (r, ix) t = Dynamic._sort (asDynamic r, longAsDynamic ix) (asDynamic t)

-- GPU only:
--   sortKeyValueInplace :: t -> IndexDynamic t -> Int -> Int -> IO ()
