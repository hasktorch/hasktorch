{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Dynamic.Tensor.Math where

import qualified Torch.Sig.Tensor.Math   as Sig
import qualified Torch.Class.Tensor.Math as Class
import qualified Torch.Types.TH as TH (IndexStorage)
import qualified Foreign.Marshal as FM

import Torch.Dimensions
import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor ()

instance Class.TensorMath Dynamic where
  _fill :: Dynamic -> HsReal -> IO ()
  _fill t v = withDynamicState t $ shuffle2 Sig.c_fill (hs2cReal v)

  _zero :: Dynamic -> IO ()
  _zero t = withDynamicState t Sig.c_zero

  _zeros :: Dynamic -> IndexStorage -> IO ()
  _zeros t ix = withDynamicState t Sig.c_zero

  _zerosLike :: Dynamic -> Dynamic -> IO ()
  _zerosLike t0 t1 = with2DynamicState t0 t1 Sig.c_zerosLike

  _ones :: Dynamic -> TH.IndexStorage -> IO ()
  _ones t0 ix = withDynamicState t0 $ \s' t0' -> withCPUIxStorage ix $ \ix' ->
    Sig.c_ones s' t0' ix'

  _onesLike :: Dynamic -> Dynamic -> IO ()
  _onesLike t0 t1 = with2DynamicState t0 t1 Sig.c_onesLike

  numel :: Dynamic -> IO Integer
  numel t = withDynamicState t (fmap fromIntegral .: Sig.c_numel)

  _reshape :: Dynamic -> Dynamic -> TH.IndexStorage -> IO ()
  _reshape t0 t1 ix = with2DynamicState t0 t1 $ \s' t0' t1' -> withCPUIxStorage ix $ \ix' ->
    Sig.c_reshape s' t0' t1' ix'

  _catArray :: Dynamic -> [Dynamic] -> Int -> DimVal -> IO ()
  _catArray res ds x y = withDynamicState res $ \s' r' -> do
    ds' <- FM.newArray =<< mapM (\d -> withForeignPtr (ctensor d) pure) ds
    Sig.c_catArray s' r' ds' (fromIntegral x) (fromIntegral y)

  _tril :: Dynamic -> Dynamic -> Integer -> IO ()
  _tril t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_tril (fromInteger i0)

  _triu :: Dynamic -> Dynamic -> Integer -> IO ()
  _triu t0 t1 i0 = with2DynamicState t0 t1 $ shuffle3 Sig.c_triu (fromInteger i0)

  _cat :: Dynamic -> Dynamic -> Dynamic -> DimVal -> IO ()
  _cat t0 t1 t2 i = withDynamicState t0 $ \s' t0' ->
    with2DynamicState t1 t2 $ \_ t1' t2' ->
      Sig.c_cat s' t0' t1' t2' (fromIntegral i)

  _nonzero :: IndexDynamic -> Dynamic -> IO ()
  _nonzero ix t = withDynamicState t $ \s' t' -> withIx ix $ \ix' -> Sig.c_nonzero s' ix' t'

  trace :: Dynamic -> IO HsAccReal
  trace t = withDynamicState t (fmap c2hsAccReal .: Sig.c_trace)

  _diag :: Dynamic -> Dynamic -> Int -> IO ()
  _diag t0 t1 i0 = with2DynamicState t0 t1 $ \s' t0' t1' -> Sig.c_diag s' t0' t1' (fromIntegral i0)

  _eye :: Dynamic -> Integer -> Integer -> IO ()
  _eye t0 l0 l1 = withDynamicState t0 $ \s' t0' -> Sig.c_eye s' t0' (fromIntegral l0) (fromIntegral l1)

  _arange :: Dynamic -> HsAccReal -> HsAccReal -> HsAccReal -> IO ()
  _arange t0 a0 a1 a2 = withDynamicState t0 $ \s' t0' -> Sig.c_arange s' t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)

  _range :: Dynamic -> HsAccReal-> HsAccReal-> HsAccReal-> IO ()
  _range t0 a0 a1 a2 = withDynamicState t0 $ \s' t0' -> Sig.c_range s' t0' (hs2cAccReal a0) (hs2cAccReal a1) (hs2cAccReal a2)


