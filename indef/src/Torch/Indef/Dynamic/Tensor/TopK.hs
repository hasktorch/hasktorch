module Torch.Indef.Dynamic.Tensor.TopK where

import qualified Torch.Sig.Tensor.TopK as Sig
import Torch.Dimensions (DimVal)

import Torch.Indef.Types

_topk :: (Dynamic, IndexDynamic) -> Dynamic -> Integer -> DimVal -> TopKOrder -> Maybe KeepDim -> IO ()
_topk (t0, ix) t1 l i0 o sorted = with2DynamicState t0 t1 $ \s' t0' t1' ->
  withIx ix $ \ix' ->
    Sig.c_topk s' t0' ix' t1' (fromIntegral l) (fromIntegral i0) (fromIntegral $ fromEnum o) (fromKeepDim sorted)

