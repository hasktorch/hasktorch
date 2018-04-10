module Torch.Indef.Static.Tensor.Masked where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Masked.Static as Class
import qualified Torch.Class.Tensor.Masked as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Masked ()

instance Class.TensorMasked Tensor where
  _maskedFill t m v = Dynamic._maskedFill (asDynamic t) (byteAsDynamic m) v
  _maskedCopy r m t = Dynamic._maskedCopy (asDynamic r) (byteAsDynamic m) (asDynamic t)
  _maskedSelect r s m = Dynamic._maskedSelect (asDynamic r) (asDynamic s) (byteAsDynamic m)


