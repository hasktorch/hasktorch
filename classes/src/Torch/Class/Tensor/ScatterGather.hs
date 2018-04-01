module Torch.Class.Tensor.ScatterGather where

import Torch.Class.Types
import Torch.Dimensions

class TensorScatterGather t where
  gather_      :: t -> t -> DimVal -> IndexDynamic t -> IO ()
  scatter_     :: t -> DimVal -> IndexDynamic t -> t -> IO ()
  scatterAdd_  :: t -> DimVal -> IndexDynamic t -> t -> IO ()
  scatterFill_ :: t -> DimVal -> IndexDynamic t -> HsReal t -> IO ()

