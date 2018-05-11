module Torch.Indef.Static.NN.Backprop where

import Numeric.Backprop
import System.IO.Unsafe
import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Pointwise.Signed ()

instance Dimensions d => Backprop (Tensor d) where
  add = (+)
  zero = (const . constant) 0
  one = (const . constant) 1


