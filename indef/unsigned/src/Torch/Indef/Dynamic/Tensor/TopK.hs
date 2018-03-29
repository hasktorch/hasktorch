module Torch.Indef.Dynamic.Tensor.TopK where

import Torch.Class.Tensor.TopK
import qualified Torch.Sig.Tensor.TopK as Sig

import Torch.Indef.Types

instance TensorTopK Dynamic where
  topk_ :: Dynamic -> IndexTensor -> Dynamic -> Integer -> Int -> Int -> Int -> IO ()
  topk_ t0 ix t1 l i0 i1 i2 = with2DynamicState t0 t1 $ \s' t0' t1' ->
    withIx ix $ \ix' ->
      Sig.c_topk s' t0' ix' t1' (fromIntegral l) (fromIntegral i0) (fromIntegral i1) (fromIntegral i2)

