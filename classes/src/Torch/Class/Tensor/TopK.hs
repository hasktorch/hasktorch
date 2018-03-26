module Torch.Class.Tensor.TopK where

import Torch.Class.Types

class TensorTopK t where
  topk_ :: t -> IndexTensor t -> t -> Integer -> Int -> Int -> Int -> io ()


