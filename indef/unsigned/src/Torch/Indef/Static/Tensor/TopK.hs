{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.TopK where

import Torch.Dimensions
import Torch.Class.Tensor.TopK
import Torch.Indef.Dynamic.Tensor.TopK ()

import Torch.Indef.Types

instance TensorTopK (Tensor d) where
  topk_ :: (Tensor d', IndexTensor) -> Tensor d -> Integer -> DimVal -> TopKOrder -> Bool -> IO ()
  topk_ (r, ri) t k d o sorted =
    topk_ (asDynamic r, ri) (asDynamic t) k d o sorted

