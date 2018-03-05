{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Tensor.Dynamic.Lapack where

import Foreign
import Foreign.C.Types
import Data.Coerce (coerce)
import Torch.Types.TH
import qualified Torch.Signature.Tensor.Lapack as Sig
import qualified Torch.Class.Tensor.Lapack as Class
import qualified Torch.Types.TH.Int as Int
import qualified Foreign.Marshal.Array as FM

import Torch.Indef.Types

instance Class.TensorLapack Tensor where
  gesv_ :: Tensor -> Tensor -> Tensor -> Tensor -> IO ()
  gesv_ = with4Tensors Sig.c_gesv

  trtrs_ :: Tensor -> Tensor -> Tensor -> Tensor -> [Int8] -> [Int8] -> [Int8] -> IO ()
  trtrs_ t0 t1 t2 t3 cs0 cs1 cs2 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    cs1' <- FM.newArray (coerce cs1 :: [CChar])
    cs2' <- FM.newArray (coerce cs2 :: [CChar])
    Sig.c_trtrs t0' t1' t2' t3' cs0' cs1' cs2'

  gels_ :: Tensor -> Tensor -> Tensor -> Tensor -> IO ()
  gels_ = with4Tensors Sig.c_gels

  syev_ :: Tensor -> Tensor -> Tensor -> [Int8] -> [Int8] -> IO ()
  syev_ t0 t1 t2 cs0 cs1 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    cs1' <- FM.newArray (coerce cs1 :: [CChar])
    Sig.c_syev t0' t1' t2' cs0' cs1'

  geev_ :: Tensor -> Tensor -> Tensor -> [Int8] -> IO ()
  geev_ t0 t1 t2 cs0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    Sig.c_geev t0' t1' t2' cs0'

  gesvd_ :: Tensor -> Tensor -> Tensor -> Tensor -> [Int8] -> IO ()
  gesvd_ t0 t1 t2 t3 cs0 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    Sig.c_gesvd t0' t1' t2' t3' cs0'

  gesvd2_ :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> [Int8] -> IO ()
  gesvd2_ t0 t1 t2 t3 t4 cs0 = _with5Tensors t0 t1 t2 t3 t4 $ \t0' t1' t2' t3' t4' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    Sig.c_gesvd2 t0' t1' t2' t3' t4' cs0'

  getri_ :: Tensor -> Tensor -> IO ()
  getri_ = with2Tensors Sig.c_getri

  potrf_ :: Tensor -> Tensor -> [Int8] -> IO ()
  potrf_ t0 t1 cs0 = _with2Tensors t0 t1 $ \t0' t1' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    Sig.c_potrf t0' t1' cs0'

  potrs_ :: Tensor -> Tensor -> Tensor -> [Int8] -> IO ()
  potrs_ t0 t1 t2 cs0 = _with3Tensors t0 t1 t2 $ \t0' t1' t2' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    Sig.c_potrs t0' t1' t2' cs0'

  potri_ :: Tensor -> Tensor -> [Int8] -> IO ()
  potri_ t0 t1 cs0 = _with2Tensors t0 t1 $ \t0' t1' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    Sig.c_potri t0' t1' cs0'

  qr_ :: Tensor -> Tensor -> Tensor -> IO ()
  qr_ = with3Tensors Sig.c_qr

  geqrf_ :: Tensor -> Tensor -> Tensor -> IO ()
  geqrf_ = with3Tensors Sig.c_geqrf

  orgqr_ :: Tensor -> Tensor -> Tensor -> IO ()
  orgqr_ = with3Tensors Sig.c_orgqr

  ormqr_ :: Tensor -> Tensor -> Tensor -> Tensor -> [Int8] -> [Int8] -> IO ()
  ormqr_ t0 t1 t2 t3 cs0 cs1 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> do
    cs0' <- FM.newArray (coerce cs0 :: [CChar])
    cs1' <- FM.newArray (coerce cs1 :: [CChar])
    Sig.c_ormqr t0' t1' t2' t3' cs0' cs1'

  pstrf_ :: Tensor -> Int.DynTensor -> Tensor -> [Int8] -> HsReal -> IO ()
  pstrf_ res it t cs0 v =
    _with2Tensors res t $ \res' t' ->
      withForeignPtr (Int.tensor it) $ \it' -> do
        cs0' <- FM.newArray (coerce cs0 :: [CChar])
        Sig.c_pstrf res' it' t' cs0' (hs2cReal v)

  btrifact_  :: Tensor -> Int.DynTensor -> Int.DynTensor -> Int32 -> Tensor -> IO ()
  btrifact_ res it0 it1 i t =
    _with2Tensors res t $ \res' t' ->
      withForeignPtr (Int.tensor it0) $ \it0' ->
        withForeignPtr (Int.tensor it1) $ \it1' ->
          Sig.c_btrifact res' it0' it1' (CInt i) t'

  btrisolve_ :: Tensor -> Tensor -> Tensor -> Int.DynTensor -> IO ()
  btrisolve_ t0 t1 t2 it =
    _with3Tensors t0 t1 t2 $ \t0' t1' t2' ->
      withForeignPtr (Int.tensor it) $ \it' ->
        Sig.c_btrisolve t0' t1' t2' it'


