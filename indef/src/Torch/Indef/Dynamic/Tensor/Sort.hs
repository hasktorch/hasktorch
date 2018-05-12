module Torch.Indef.Dynamic.Tensor.Sort where

import Torch.Indef.Types
import Torch.Dimensions
import qualified Torch.Sig.Tensor.Sort as Sig

_sort :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> SortOrder -> IO ()
_sort (r, ix) t1 i0 i1 = with2DynamicState r t1 $ \s' r' t1' ->
  withIx ix $ \ix' ->
    Sig.c_sort s' r' ix' t1' (fromIntegral i0) (fromIntegral $ fromEnum i1)


-- THC Only:
-- sortKeyValueInplace :: t -> IndexDynamic t -> Int -> Int -> IO ()
