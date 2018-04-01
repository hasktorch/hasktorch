module Torch.Indef.Static.Tensor.Math.Floating where

import GHC.Int
import Torch.Dimensions
import qualified Torch.Class.Tensor.Math as Dynamic
import qualified Torch.Class.Tensor.Math.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Floating ()

instance Class.TensorMathFloating Tensor where
  linspace_ :: Dimensions d => Tensor d -> HsReal -> HsReal -> Int64 -> IO ()
  linspace_ r = Dynamic.linspace_ (asDynamic r)

  logspace_ :: Dimensions d => Tensor d -> HsReal -> HsReal -> Int64 -> IO ()
  logspace_ r = Dynamic.logspace_ (asDynamic r)

