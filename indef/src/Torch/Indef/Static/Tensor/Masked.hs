module Torch.Indef.Static.Tensor.Masked where


import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Masked as Dynamic

_maskedFill t m v = Dynamic._maskedFill (asDynamic t) (byteAsDynamic m) v
_maskedCopy r m t = Dynamic._maskedCopy (asDynamic r) (byteAsDynamic m) (asDynamic t)
_maskedSelect r s m = Dynamic._maskedSelect (asDynamic r) (asDynamic s) (byteAsDynamic m)


