module Torch.Class.Tensor.Mode where

import Torch.Class.Types

class TensorMode t where
  mode_ :: (t, IndexTensor t) -> t -> Int -> Int -> io ()

