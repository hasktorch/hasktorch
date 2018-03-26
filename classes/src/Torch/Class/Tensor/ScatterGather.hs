module Torch.Class.Tensor.ScatterGather where

import Torch.Class.Types

class TensorGatherScatter t where
  gather_      :: t -> t -> Int -> IndexTensor t -> io ()
  scatter_     :: t -> Int -> IndexTensor t -> t -> io ()
  scatterAdd_  :: t -> Int -> IndexTensor t -> t -> io ()
  scatterFill_ :: t -> Int -> IndexTensor t -> HsReal t -> io ()

