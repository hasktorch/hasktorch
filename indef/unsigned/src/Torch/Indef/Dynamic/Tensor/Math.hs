{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Dynamic.Tensor.Math where

import qualified Torch.Sig.Tensor.Math   as Sig
import qualified Torch.Class.Tensor.Math as Class
import qualified Torch.Types.TH as TH (IndexStorage)
import qualified Foreign.Marshal as FM

import Torch.Dimensions
import Torch.Indef.Types

instance Class.TensorMath Dynamic where
  fill_ :: Dynamic -> HsReal -> IO ()
  fill_ t v = withDynamicState t $ shuffle2 Sig.c_fill (hs2cReal v)

  zero_ :: Dynamic -> IO ()
  zero_ t = withDynamicState t Sig.c_zero

  zeros_ :: Dynamic -> IndexStorage -> IO ()
  zeros_ t ix = withDynamicState t Sig.c_zero

  zerosLike_ :: Dynamic -> Dynamic -> IO ()
  zerosLike_ t0 t1 = with2DynamicState t0 t1 Sig.c_zerosLike

  ones_ :: Dynamic -> TH.IndexStorage -> IO ()
  ones_ t0 ix = withDynamicState t0 $ \s' t0' -> withCPUIxStorage ix $ \ix' ->
    Sig.c_ones s' t0' ix'

  onesLike_ :: Dynamic -> Dynamic -> IO ()
  onesLike_ t0 t1 = with2DynamicState t0 t1 Sig.c_onesLike

  numel :: Dynamic -> IO Integer
  numel t = withDynamicState t (fmap fromIntegral .: Sig.c_numel)

  reshape_ :: Dynamic -> Dynamic -> TH.IndexStorage -> IO ()
  reshape_ t0 t1 ix = with2DynamicState t0 t1 $ \s' t0' t1' -> withCPUIxStorage ix $ \ix' ->
    Sig.c_reshape s' t0' t1' ix'

  catArray_ :: Dynamic -> [Dynamic] -> Int -> DimVal -> IO ()
  catArray_ res ds x y = withDynamicState res $ \s' r' -> do
    ds' <- FM.newArray =<< mapM (\d -> withForeignPtr (ctensor d) pure) ds
    Sig.c_catArray s' r' ds' (fromIntegral x) (fromIntegral y)

  tril_ :: Dynamic -> Dynamic -> Integer -> IO ()
  tril_ t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_tril (fromInteger i0)

  triu_ :: Dynamic -> Dynamic -> Integer -> IO ()
  triu_ t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_triu (fromInteger i0)

  cat_ :: Dynamic -> Dynamic -> Dynamic -> DimVal -> IO ()
  cat_ t0 t1 t2 i = withDynamicState t0 $ \s' t0' ->
    with2DynamicState t1 t2 $ \_ t1' t2' ->
      Sig.c_cat s' t0' t1' t2' (fromIntegral i)

  nonzero_ :: IndexDynamic -> Dynamic -> IO ()
  nonzero_ ix t = withDynamicState t $ \s' t' -> withIx ix $ \ix' -> Sig.c_nonzero s' ix' t'

  trace :: Dynamic -> IO HsAccReal
  trace t = withDynamicState t (fmap c2hsAccReal .: Sig.c_trace)

  diag_ :: Dynamic -> Dynamic -> Int -> IO ()
  diag_ t0 t1 i0 = with2DynamicState t0 t1 $ \s' t0' t1' -> Sig.c_diag s' t0' t1' (fromIntegral i0)

  eye_ :: Dynamic -> Integer -> Integer -> IO ()
  eye_ t0 l0 l1 = withDynamicState t0 $ \s' t0' -> Sig.c_eye s' t0' (fromIntegral l0) (fromIntegral l1)

  arange_ :: Dynamic -> HsAccReal -> HsAccReal -> HsAccReal -> IO ()
  arange_ t0 a0 a1 a2 = withDynamicState t0 $ \s' t0' -> Sig.c_arange s' t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

  range_ :: Dynamic -> HsAccReal-> HsAccReal-> HsAccReal-> IO ()
  range_ t0 a0 a1 a2 = withDynamicState t0 $ \s' t0' -> Sig.c_range s' t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)


