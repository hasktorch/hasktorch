module Torch.Core.Tensor.Static.Copy where

import SigTypes
import Torch.Core.Tensor.Dynamic.Copy ()
import qualified Torch.Class.C.Tensor.Copy as Class

instance Class.TensorCopy (Tensor d) where
  copy t = asStatic <$> Class.copy (dynamic t)
  copyByte t = Class.copyByte (dynamic t)
  copyShort t = Class.copyShort (dynamic t)
  copyInt t = Class.copyInt (dynamic t)
  copyLong t = Class.copyLong (dynamic t)
  copyFloat t = Class.copyFloat (dynamic t)
  copyDouble t = Class.copyDouble (dynamic t)


