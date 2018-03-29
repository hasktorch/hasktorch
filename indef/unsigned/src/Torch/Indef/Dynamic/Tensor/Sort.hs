module Torch.Indef.Dynamic.Tensor.Sort where

import Torch.Class.Tensor.Sort
import Torch.Indef.Types
import qualified Torch.Sig.Tensor.Sort as Sig

instance TensorSort Dynamic where
  sort_ :: Dynamic -> IndexTensor -> Dynamic -> Int -> Int -> IO ()
  sort_ t0 ix t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
    withIx ix $ \ix' ->
      Sig.c_sort s' t0' ix' t1' (fromIntegral i0) (fromIntegral i1)


