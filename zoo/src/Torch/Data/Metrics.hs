module Torch.Data.Metrics where

import Data.List (genericLength)
import Torch.Double
import qualified Torch.Long as Long

accuracy :: [(Tensor '[10], Integer)] -> Double
accuracy xs = foldl go 0 xs / genericLength xs
  where
    go :: Double -> (Tensor '[10], Integer) -> Double
    go acc (p, y) = acc + fromIntegral (fromEnum (y == fromIntegral (Long.get1d (maxIndex1d p) 0)))


