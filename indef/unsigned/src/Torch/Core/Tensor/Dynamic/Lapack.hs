{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Lapack where

import Foreign
import Foreign.C.Types
import qualified TensorLapack as Sig
import qualified Torch.Class.Tensor.Lapack as Class
import THTypes

import Torch.Core.Types
{-
instance Class.TensorLapack Tensor where
  gesv :: Tensor -> Tensor -> Tensor -> Tensor -> IO ()
  gesv = with4Tensors Sig.c_gesv

  trtrs :: Tensor -> Tensor -> Tensor -> Tensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()
  trtrs t0 t1 t2 t3 cs0 cs1 cs2 = _with4Tensors t0 t1 t2 t3 (\t0' t1' t2' t3' -> Sig.c_trtrs t0' t1' t2' t3' cs0 cs1 cs2)

  gels :: Tensor -> Tensor -> Tensor -> Tensor -> IO ()
  gels = with4Tensors Sig.c_gels

  syev :: Tensor -> Tensor -> Tensor -> Ptr CChar -> Ptr CChar -> IO ()
  syev t0 t1 t2 cs0 cs1 = _with3Tensors t0 t1 t2 (\t0' t1' t2' -> Sig.c_syev t0' t1' t2' cs0 cs1)

  geev :: Tensor -> Tensor -> Tensor -> Ptr CChar -> IO ()
  geev t0 t1 t2 cs0 = _with3Tensors t0 t1 t2 (\t0' t1' t2' -> Sig.c_geev t0' t1' t2' cs0)

  gesvd :: Tensor -> Tensor -> Tensor -> Tensor -> Ptr CChar -> IO ()
  gesvd t0 t1 t2 t3 cs0 = _with4Tensors t0 t1 t2 t3 (\t0' t1' t2' t3' -> Sig.c_gesvd t0' t1' t2' t3' cs0)

  gesvd2 :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> Ptr CChar -> IO ()
  gesvd2 t0 t1 t2 t3 t4 cs0 = _with5Tensors t0 t1 t2 t3 t4 (\t0' t1' t2' t3' t4' -> Sig.c_gesvd2 t0' t1' t2' t3' t4' cs0)

  getri :: Tensor -> Tensor -> IO ()
  getri = with2Tensors Sig.c_getri

  potrf :: Tensor -> Tensor -> Ptr CChar -> IO ()
  potrf t0 t1 cs0 = _with2Tensors t0 t1 (\t0' t1' -> Sig.c_potrf t0' t1' cs0)

  potrs :: Tensor -> Tensor -> Tensor -> Ptr CChar -> IO ()
  potrs t0 t1 t2 cs0 = _with3Tensors t0 t1 t2 (\t0' t1' t2' -> Sig.c_potrs t0' t1' t2' cs0)

  potri :: Tensor -> Tensor -> Ptr CChar -> IO ()
  potri t0 t1 cs0 = _with2Tensors t0 t1 (\t0' t1' -> Sig.c_potri t0' t1' cs0)

  qr :: Tensor -> Tensor -> Tensor -> IO ()
  qr = with3Tensors Sig.c_qr

  geqrf :: Tensor -> Tensor -> Tensor -> IO ()
  geqrf = with3Tensors Sig.c_geqrf

  orgqr :: Tensor -> Tensor -> Tensor -> IO ()
  orgqr = with3Tensors Sig.c_orgqr

  ormqr :: Tensor -> Tensor -> Tensor -> Tensor -> Ptr CChar -> Ptr CChar -> IO ()
  ormqr t0 t1 t2 t3 cs0 cs1 = _with4Tensors t0 t1 t2 t3 (\t0' t1' t2' t3' -> Sig.c_ormqr t0' t1' t2' t3' cs0 cs1)

  pstrf :: Tensor -> Ptr CTHIntTensor -> Tensor -> Ptr CChar -> HsReal -> IO ()
  pstrf res it t cs0 v = _with2Tensors res t (\res' t' -> Sig.c_pstrf res' it t' cs0 (hs2cReal v))

  btrifact  :: Tensor -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> Int32 -> Tensor -> IO ()
  btrifact res it0 it1 i t = _with2Tensors res t (\res' t' -> Sig.c_btrifact res' it0 it1 (CInt i) t')

  btrisolve :: Tensor -> Tensor -> Tensor -> Ptr CTHIntTensor -> IO ()
  btrisolve t0 t1 t2 it = _with3Tensors t0 t1 t2 (\t0' t1' t2' -> Sig.c_btrisolve t0' t1' t2' it)
  -}
