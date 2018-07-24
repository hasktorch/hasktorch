module Torch.Data.Metrics where

import Data.List (genericLength)

#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double
import qualified Torch.Long as Long
#endif

accuracy :: [(Tensor '[10], Integer)] -> Double
accuracy xs = foldl go 0 xs / genericLength xs
  where
    go :: Double -> (Tensor '[10], Integer) -> Double
    go acc (p, y) = acc + fromIntegral (fromEnum (y == fromIntegral (Long.get1d (maxIndex1d p) 0)))


