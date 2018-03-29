module Torch.Indef.Dynamic.Tensor.TopK where

import Torch.Class.Tensor.TopK
import qualified Torch.Sig.Tensor.TopK as Sig
import Torch.Dimensions (DimVal)

import Torch.Indef.Types

instance TensorTopK Dynamic where
  topk_ :: (Dynamic, IndexDynamic) -> Dynamic -> Integer -> DimVal -> TopKOrder -> Bool -> IO ()
  topk_ (t0, ix) t1 l i0 o sorted = with2DynamicState t0 t1 $ \s' t0' t1' ->
    withIx ix $ \ix' ->
      Sig.c_topk s' t0' ix' t1' (fromIntegral l) (fromIntegral i0) (fromIntegral $ fromEnum o) (fromIntegral $ fromEnum sorted)

