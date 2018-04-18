module Torch.Class.Tensor.Index where

import Data.Word
import Torch.Class.Types

class TensorIndex t where
  _indexCopy :: t -> Int -> IndexDynamic t -> t -> IO ()
  _indexAdd :: t -> Int -> IndexDynamic t -> t -> IO ()
  _indexFill :: t -> Int -> IndexDynamic t -> HsReal t -> IO ()
  _indexSelect :: t -> t -> Int -> IndexDynamic t -> IO ()
  _take :: t -> t -> IndexDynamic t -> IO ()
  _put :: t -> IndexDynamic t -> t -> Int -> IO ()

class GPUTensorIndex t where
  _indexCopy_long :: t -> Int -> IndexDynamic t -> t -> IO ()
  _indexAdd_long :: t -> Int -> IndexDynamic t -> t -> IO ()
  _indexFill_long :: t -> Int -> IndexDynamic t -> Word -> IO ()
  _indexSelect_long :: t -> t -> Int -> IndexDynamic t -> IO ()
  _calculateAdvancedIndexingOffsets :: IndexDynamic t -> t -> Integer -> [IndexDynamic t] -> IO ()
