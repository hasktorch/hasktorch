module Torch.Indef.Static.Tensor.Mode where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Mode.Static as Class
import qualified Torch.Class.Tensor.Mode as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Mode ()

instance Class.TensorMode Tensor where
  mode_ :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  mode_ (r, ix) t = Dynamic.mode_ (asDynamic r, longAsDynamic ix) (asDynamic t)

