module Torch.Class.Tensor.Sort where

import Torch.Class.Types

class TensorSort t where
  sort_ :: t -> IndexTensor t -> t -> Int -> Int -> IO ()

class GPUTensorSort t where
  sortKeyValueInplace :: t -> IndexTensor t -> Int -> Int -> IO ()
