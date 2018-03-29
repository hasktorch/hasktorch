module Torch.Indef.Dynamic.Tensor.ScatterGather where

import Torch.Class.Tensor.ScatterGather
import Torch.Indef.Types
import qualified Torch.Sig.Tensor.ScatterGather as Sig

instance TensorGatherScatter Dynamic where
  gather_ :: Dynamic -> Dynamic -> Int -> IndexTensor -> IO ()
  gather_ t0 t1 i ix = with2DynamicState t0 t1 $ \s' t0' t1' -> withIx ix $ \ix' ->
    Sig.c_gather s' t0' t1' (fromIntegral i) ix'

  scatter_ :: Dynamic -> Int -> IndexTensor -> Dynamic -> IO ()
  scatter_ t0 i ix t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withIx ix $ \ix' ->
    Sig.c_scatter s' t0' (fromIntegral i) ix' t1'

  scatterAdd_   :: Dynamic -> Int -> IndexTensor -> Dynamic -> IO ()
  scatterAdd_ t0 i ix t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withIx ix $ \ix' ->
    Sig.c_scatterAdd s' t0' (fromIntegral i) ix' t1'

  scatterFill_  :: Dynamic -> Int -> IndexTensor -> HsReal -> IO ()
  scatterFill_ t0 i ix v = withDynamicState t0 $ \s' t0' -> withIx ix $ \ix' ->
    Sig.c_scatterFill s' t0' (fromIntegral i) ix' (hs2cReal v)
