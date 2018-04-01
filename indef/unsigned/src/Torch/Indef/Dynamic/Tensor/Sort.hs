module Torch.Indef.Dynamic.Tensor.Sort where

import Torch.Class.Tensor.Sort
import Torch.Indef.Types
import Torch.Dimensions
import qualified Torch.Sig.Tensor.Sort as Sig

instance TensorSort Dynamic where
  sort_ :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> SortOrder -> IO ()
  sort_ (r, ix) t1 i0 i1 = with2DynamicState r t1 $ \s' r' t1' ->
    withIx ix $ \ix' ->
      Sig.c_sort s' r' ix' t1' (fromIntegral i0) (fromIntegral $ fromEnum i1)


