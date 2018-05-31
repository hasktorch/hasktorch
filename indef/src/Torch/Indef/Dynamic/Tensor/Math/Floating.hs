module Torch.Indef.Dynamic.Tensor.Math.Floating where

import GHC.Int
import qualified Torch.Sig.Tensor.Math.Floating as Sig

import Torch.Indef.Types

_linspace :: Dynamic -> HsReal -> HsReal -> Int64 -> IO ()
_linspace r a b l = withDynamicState r $ \s' r' -> Sig.c_linspace s' r' (hs2cReal a) (hs2cReal b) (fromIntegral l)

_logspace :: Dynamic -> HsReal -> HsReal -> Int64 -> IO ()
_logspace r a b l = withDynamicState r $ \s' r' -> Sig.c_logspace s' r' (hs2cReal a) (hs2cReal b) (fromIntegral l)

