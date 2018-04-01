module Torch.Indef.Static.Tensor.Masked where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Masked.Static as Class
import qualified Torch.Class.Tensor.Masked as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Masked ()

instance Class.TensorMasked Tensor where
  maskedFill_ t m v = Dynamic.maskedFill_ (asDynamic t) (byteAsDynamic m) v
  maskedCopy_ r m t = Dynamic.maskedCopy_ (asDynamic r) (byteAsDynamic m) (asDynamic t)
  maskedSelect_ r s m = Dynamic.maskedSelect_ (asDynamic r) (asDynamic s) (byteAsDynamic m)


