module Torch.Indef.Static.Tensor.Math.Floating where

import GHC.Int

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Floating as Dynamic

_linspace :: Dimensions d => Tensor d -> HsReal -> HsReal -> Int64 -> IO ()
_linspace r = Dynamic._linspace (asDynamic r)

_logspace :: Dimensions d => Tensor d -> HsReal -> HsReal -> Int64 -> IO ()
_logspace r = Dynamic._logspace (asDynamic r)

