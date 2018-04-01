module Torch.Indef.Static.Tensor.TopK where

import Torch.Dimensions
import qualified Torch.Class.Tensor.TopK.Static as Class
import Torch.Class.Tensor.TopK as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.TopK ()

instance Class.TensorTopK Tensor where
  topk_ :: (Tensor d', IndexTensor d') -> Tensor d -> Integer -> DimVal -> TopKOrder -> Maybe KeepDim -> IO ()
  topk_ (r, ix) t = Dynamic.topk_ (asDynamic r, longAsDynamic ix) (asDynamic t)

