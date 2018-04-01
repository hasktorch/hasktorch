module Torch.Indef.Static.Tensor.Copy where

import Torch.Types.TH
import Torch.Indef.Types (Tensor, asDynamic, asStatic)
import Torch.Indef.Dynamic.Tensor.Copy ()
import Torch.Dimensions
import qualified Torch.Class.Tensor.Copy.Static as Class
import qualified Torch.Class.Tensor.Copy as Dynamic

instance Class.TensorCopy Tensor where
  copy t = asStatic <$> Dynamic.copy (asDynamic t)
  copyByte t = byteAsStatic <$> Dynamic.copyByte (asDynamic t)
  copyShort t = shortAsStatic <$> Dynamic.copyShort (asDynamic t)
  copyInt t = intAsStatic <$> Dynamic.copyInt (asDynamic t)
  copyLong t = longAsStatic <$> Dynamic.copyLong (asDynamic t)
  copyFloat t = floatAsStatic <$> Dynamic.copyFloat (asDynamic t)
  copyDouble t = doubleAsStatic <$> Dynamic.copyDouble (asDynamic t)


