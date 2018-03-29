module Torch.Class.Tensor.ScatterGather where

import Torch.Class.Types

class TensorGatherScatter t where
  gather_      :: t -> t -> Int -> IndexTensor t -> IO ()
  scatter_     :: t -> Int -> IndexTensor t -> t -> IO ()
  scatterAdd_  :: t -> Int -> IndexTensor t -> t -> IO ()
  scatterFill_ :: t -> Int -> IndexTensor t -> HsReal t -> IO ()

