module Torch.Class.Tensor.Index where

import Data.Word
import Torch.Class.Types

class TensorIndex t where
  indexCopy_ :: t -> Int -> IndexDynamic t -> t -> IO ()
  indexAdd_ :: t -> Int -> IndexDynamic t -> t -> IO ()
  indexFill_ :: t -> Int -> IndexDynamic t -> HsReal t -> IO ()
  indexSelect_ :: t -> t -> Int -> IndexDynamic t -> IO ()
  take_ :: t -> t -> IndexDynamic t -> IO ()
  put_ :: t -> IndexDynamic t -> t -> Int -> IO ()

class GPUTensorIndex t where
  indexCopy_long_ :: t -> Int -> IndexDynamic t -> t -> IO ()
  indexAdd_long_ :: t -> Int -> IndexDynamic t -> t -> IO ()
  indexFill_long_ :: t -> Int -> IndexDynamic t -> Word -> IO ()
  indexSelect_long_ :: t -> t -> Int -> IndexDynamic t -> IO ()
  calculateAdvancedIndexingOffsets_ :: IndexDynamic t -> t -> Integer -> [IndexDynamic t] -> IO ()
