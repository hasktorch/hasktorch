module Torch.Indef.Static.NN.Backprop where

import Numeric.Backprop
import System.IO.Unsafe
import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Pointwise.Signed ()
import qualified Torch.Indef.Index as Ix

instance Dimensions d => Backprop (Tensor d) where
  add = (+)
  zero = (const . constant) 0
  one = (const . constant) 1

-- FIXME: find out what to do with this
instance Dimensions d => Backprop (IndexTensor d) where
  add a b = b
  zero = const Ix.zeroIxNd
  one = const (longAsStatic (Ix.newIxDyn 1))

