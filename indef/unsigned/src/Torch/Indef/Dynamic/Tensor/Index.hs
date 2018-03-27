module Torch.Indef.Dynamic.Tensor.Index where

import Torch.Indef.Internal
import Torch.Class.Dynamic.Tensor.Index

instance TensorIndex DynTensor where
  indexCopy_ :: t -> Int -> IndexTensor t -> t -> IO ()
  indexAdd_ :: t -> Int -> IndexTensor t -> t -> IO ()
  indexFill_ :: t -> Int -> IndexTensor t -> HsReal t -> IO ()
  indexSelect_ :: t -> t -> Int -> IndexTensor t -> IO ()
  take_ :: t -> t -> IndexTensor t -> IO ()
  put_ :: t -> IndexTensor t -> t -> Int -> IO ()

lass GPUTensorIndex t where
  indexCopy_long_ :: t -> Int -> IndexTensor t -> t -> IO ()
  indexAdd_long_ :: t -> Int -> IndexTensor t -> t -> IO ()
  indexFill_long_ :: t -> Int -> IndexTensor t -> Word -> IO ()
  indexSelect_long_ :: t -> t -> Int -> IndexTensor t -> IO ()
  calculateAdvancedIndexingOffsets_ :: IndexTensor t -> t -> Integer -> [IndexTensor t] -> IO ()
