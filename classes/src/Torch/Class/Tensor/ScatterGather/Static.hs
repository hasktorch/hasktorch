module Torch.Class.Tensor.ScatterGather.Static where

import Torch.Class.Types
import Torch.Dimensions

class TensorScatterGather t where
  gather_      :: Dimensions d => t d -> t d -> DimVal -> IndexTensor (t d) '[n] -> IO ()
  scatter_     :: Dimensions d => t d -> DimVal -> IndexTensor (t d) '[n] -> t d -> IO ()
  scatterAdd_  :: Dimensions d => t d -> DimVal -> IndexTensor (t d) '[n] -> t d -> IO ()
  scatterFill_ :: Dimensions d => t d -> DimVal -> IndexTensor (t d) '[n] -> HsReal (t d) -> IO ()

