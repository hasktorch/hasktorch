module Torch.Indef.Dynamic.Tensor.Math.Reduce where

import Torch.Class.Tensor.Math.Reduce
import Torch.Indef.Types

import qualified Torch.Sig.Tensor.Math.Reduce as Sig

instance TensorMathReduce Dynamic where
  minall :: Dynamic -> IO HsReal
  minall = flip withDynamicState (fmap c2hsReal .: Sig.c_minall)

  maxall :: Dynamic -> IO HsReal
  maxall = flip withDynamicState (fmap c2hsReal .: Sig.c_maxall)

  medianall :: Dynamic -> IO HsReal
  medianall = flip withDynamicState (fmap c2hsReal .: Sig.c_medianall)

  sumall :: Dynamic -> IO HsAccReal
  sumall = flip withDynamicState (fmap c2hsAccReal .: Sig.c_sumall)

  prodall :: Dynamic -> IO HsAccReal
  prodall = flip withDynamicState (fmap c2hsAccReal .: Sig.c_prodall)

  max_ :: (Dynamic, IndexDynamic) -> Dynamic -> Int -> Int -> IO ()
  max_ (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
    withIx ix $ \ix' ->
      Sig.c_max s' t0' ix' t1' (fromIntegral i0) (fromIntegral i1)

  min_ :: (Dynamic, IndexDynamic) -> Dynamic -> Int -> Int -> IO ()
  min_ (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
    withIx ix $ \ix' ->
      Sig.c_min s' t0' ix' t1' (fromIntegral i0) (fromIntegral i1)

  median_ :: (Dynamic, IndexDynamic) -> Dynamic -> Int -> Int -> IO ()
  median_ (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
    withIx ix $ \ix' ->
      Sig.c_median s' t0' ix' t1' (fromIntegral i0) (fromIntegral i1)

  sum_ :: Dynamic -> Dynamic -> Int -> Int -> IO ()
  sum_ t0 t1 i0 i1 = with2DynamicState t0 t1 $ shuffle3'2 Sig.c_sum (fromIntegral i0) (fromIntegral i1)

  prod_ :: Dynamic -> Dynamic -> Int -> Int -> IO ()
  prod_ t0 t1 i0 i1 = with2DynamicState t0 t1 $ shuffle3'2 Sig.c_prod (fromIntegral i0) (fromIntegral i1)


