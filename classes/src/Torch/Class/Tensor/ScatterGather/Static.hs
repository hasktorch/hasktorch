module Torch.Class.Tensor.ScatterGather.Static where

import Torch.Class.Types
import Torch.Dimensions

class TensorScatterGather t where
  _gather      :: Dimensions d => t d -> t d -> DimVal -> IndexTensor t '[n] -> IO ()
  _scatter     :: Dimensions d => t d -> DimVal -> IndexTensor t '[n] -> t d -> IO ()
  _scatterAdd  :: Dimensions d => t d -> DimVal -> IndexTensor t '[n] -> t d -> IO ()
  _scatterFill :: Dimensions d => t d -> DimVal -> IndexTensor t '[n] -> HsReal (t d) -> IO ()

