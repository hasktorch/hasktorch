-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Backprop
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Backprop helper instances for static tensors, as well as any helper
-- functions that might work well with backprop.
-------------------------------------------------------------------------------
module Torch.Indef.Static.NN.Backprop where

import Numeric.Backprop
import Numeric.Dimensions

import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Pointwise.Signed ()
import qualified Torch.Indef.Index as Ix

instance Dimensions d => Backprop (Tensor d) where
  add = (+)
  zero = (const . constant) 0
  one = (const . constant) 1

