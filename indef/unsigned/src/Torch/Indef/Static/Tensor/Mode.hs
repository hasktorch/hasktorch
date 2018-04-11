module Torch.Indef.Static.Tensor.Mode where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Mode.Static as Class
import qualified Torch.Class.Tensor.Mode as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Mode ()

instance Class.TensorMode Tensor where
  _mode :: (Tensor d, IndexTensor '[n]) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  _mode (r, ix) t = Dynamic._mode (asDynamic r, longAsDynamic ix) (asDynamic t)

