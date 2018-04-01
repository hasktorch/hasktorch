module Torch.Indef.Dynamic.Tensor.ScatterGather where

import Torch.Class.Tensor.ScatterGather
import Torch.Indef.Types
import Torch.Dimensions
import qualified Torch.Sig.Tensor.ScatterGather as Sig

instance TensorScatterGather Dynamic where
  gather_ :: Dynamic -> Dynamic -> DimVal -> IndexDynamic -> IO ()
  gather_ r src d ix = with2DynamicState r src $ \s' r' src' -> withIx ix $ \ix' ->
    Sig.c_gather s' r' src' (fromIntegral d) ix'

  scatter_ :: Dynamic -> DimVal -> IndexDynamic -> Dynamic -> IO ()
  scatter_ r d ix src = with2DynamicState r src $ \s' r' src' -> withIx ix $ \ix' ->
    Sig.c_scatter s' r' (fromIntegral d) ix' src'

  scatterAdd_   :: Dynamic -> DimVal -> IndexDynamic -> Dynamic -> IO ()
  scatterAdd_ r d ix src = with2DynamicState r src $ \s' r' src' -> withIx ix $ \ix' ->
    Sig.c_scatterAdd s' r' (fromIntegral d) ix' src'

  scatterFill_  :: Dynamic -> DimVal -> IndexDynamic -> HsReal -> IO ()
  scatterFill_ r d ix v = withDynamicState r $ \s' r' -> withIx ix $ \ix' ->
    Sig.c_scatterFill s' r' (fromIntegral d) ix' (hs2cReal v)
