module Torch.Indef.Dynamic.Tensor.ScatterGather where

import Torch.Class.Tensor.ScatterGather
import Torch.Indef.Types
import Torch.Dimensions
import qualified Torch.Sig.Tensor.ScatterGather as Sig

instance TensorScatterGather Dynamic where
  _gather :: Dynamic -> Dynamic -> DimVal -> IndexDynamic -> IO ()
  _gather r src d ix = with2DynamicState r src $ \s' r' src' -> withIx ix $ \ix' ->
    Sig.c_gather s' r' src' (fromIntegral d) ix'

  _scatter :: Dynamic -> DimVal -> IndexDynamic -> Dynamic -> IO ()
  _scatter r d ix src = with2DynamicState r src $ \s' r' src' -> withIx ix $ \ix' ->
    Sig.c_scatter s' r' (fromIntegral d) ix' src'

  _scatterAdd   :: Dynamic -> DimVal -> IndexDynamic -> Dynamic -> IO ()
  _scatterAdd r d ix src = with2DynamicState r src $ \s' r' src' -> withIx ix $ \ix' ->
    Sig.c_scatterAdd s' r' (fromIntegral d) ix' src'

  _scatterFill  :: Dynamic -> DimVal -> IndexDynamic -> HsReal -> IO ()
  _scatterFill r d ix v = withDynamicState r $ \s' r' -> withIx ix $ \ix' ->
    Sig.c_scatterFill s' r' (fromIntegral d) ix' (hs2cReal v)
