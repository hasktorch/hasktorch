module Torch.Indef.Static.Tensor.TopK where

import Torch.Dimensions
import qualified Torch.Class.Tensor.TopK.Static as Class
import Torch.Class.Tensor.TopK as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor ()
import Torch.Indef.Dynamic.Tensor.TopK ()
import Torch.Indef.Static.Tensor ()

instance Class.TensorTopK Tensor where
  _topk :: (Tensor d', IndexTensor d') -> Tensor d -> Integer -> DimVal -> TopKOrder -> Maybe KeepDim -> IO ()
  _topk (r, ix) t = Dynamic._topk (asDynamic r, longAsDynamic ix) (asDynamic t)

