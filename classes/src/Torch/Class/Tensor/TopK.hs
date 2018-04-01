{-# LANGUAGE TypeFamilies #-}
module Torch.Class.Tensor.TopK where

import Torch.Class.Types
import Torch.Dimensions
import Torch.Class.Tensor

-- https://github.com/torch/torch7/blob/75a86469aa9e2f5f04e11895b269ec22eb0e4687/lib/TH/generic/THTensorMath.c#L2545
data TopKOrder = KAscending | KNone | KDescending
  deriving (Eq, Show, Ord, Enum, Bounded)

class TensorTopK t where
  topk_ :: (t, IndexDynamic t) -> t -> Integer -> DimVal -> TopKOrder -> Maybe KeepDim -> IO ()

