module Torch.Class.Tensor.ScatterGather where

import Torch.Class.Types
import Torch.Dimensions

class TensorScatterGather t where
  _gather      :: t -> t -> DimVal -> IndexDynamic t -> IO ()
  _scatter     :: t -> DimVal -> IndexDynamic t -> t -> IO ()
  _scatterAdd  :: t -> DimVal -> IndexDynamic t -> t -> IO ()
  _scatterFill :: t -> DimVal -> IndexDynamic t -> HsReal t -> IO ()

