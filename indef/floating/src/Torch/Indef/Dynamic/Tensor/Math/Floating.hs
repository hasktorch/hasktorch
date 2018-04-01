module Torch.Indef.Dynamic.Tensor.Math.Floating where

import GHC.Int
import qualified Torch.Class.Tensor.Math as Class
import qualified Torch.Sig.Tensor.Math.Floating as Sig

import Torch.Indef.Types

instance Class.TensorMathFloating Dynamic where
  linspace_ :: Dynamic -> HsReal -> HsReal -> Int64 -> IO ()
  linspace_ r a b l = withDynamicState r $ \s' r' -> Sig.c_linspace s' r' (hs2cReal a) (hs2cReal b) (fromIntegral l)

  logspace_ :: Dynamic -> HsReal -> HsReal -> Int64 -> IO ()
  logspace_ r a b l = withDynamicState r $ \s' r' -> Sig.c_logspace s' r' (hs2cReal a) (hs2cReal b) (fromIntegral l)

