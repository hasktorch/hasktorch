module Torch.Indef.Static.Tensor.Math.Floating where

import GHC.Int
import Torch.Dimensions
import qualified Torch.Class.Tensor.Math as Dynamic
import qualified Torch.Class.Tensor.Math.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Floating ()

instance Class.TensorMathFloating Tensor where
  _linspace :: Dimensions d => Tensor d -> HsReal -> HsReal -> Int64 -> IO ()
  _linspace r = Dynamic._linspace (asDynamic r)

  _logspace :: Dimensions d => Tensor d -> HsReal -> HsReal -> Int64 -> IO ()
  _logspace r = Dynamic._logspace (asDynamic r)

