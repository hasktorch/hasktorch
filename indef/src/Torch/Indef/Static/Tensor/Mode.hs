module Torch.Indef.Static.Tensor.Mode where


import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Mode as Dynamic

_mode :: (Tensor d, IndexTensor '[n]) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
_mode (r, ix) t = Dynamic._mode (asDynamic r, longAsDynamic ix) (asDynamic t)

