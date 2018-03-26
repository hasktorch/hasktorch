module Torch.Class.Tensor.Sort where

import Torch.Class.Types

class TensorSort t where
  sort_ :: t -> IndexTensor t -> t -> Int -> Int -> io ()

class GPUTensorSort t where
  sortKeyValueInplace :: t -> IndexTensor t -> Int -> Int -> IO ()
