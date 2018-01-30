{-# LANGUAGE TypeSynonymInstances #-}

module Torch.Raw.Lapack
  ( THLapack(..)
  , module X
  ) where

import Torch.Raw.Internal as X

import qualified THLongLapack as T
import qualified THIntLapack as T
import qualified THShortLapack as T
-- import qualified THHalfLapack as T
import qualified THByteLapack as T
import qualified THDoubleLapack as T
import qualified THFloatLapack as T


class THLapack t where
  c_gesv  :: CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_gels  :: CChar -> CInt -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_syev  :: CChar -> CChar -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_geev  :: CChar -> CChar -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_getrf :: CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> CIntPtr -> IO ()
  c_getrs :: CChar -> CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_getri :: CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ()
  p_gesv  :: FunPtr (CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_trtrs :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_gels  :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_syev  :: FunPtr (CChar -> CChar -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_geev  :: FunPtr (CChar -> CChar -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_gesvd :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_getrf :: FunPtr (CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> CIntPtr -> IO ())
  p_getrs :: FunPtr (CChar -> CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_getri :: FunPtr (CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ())

instance THLapack THFloatLapack where
  c_gesv  = T.c_THFloatLapack_gesv
  c_trtrs = T.c_THFloatLapack_trtrs
  c_gels  = T.c_THFloatLapack_gels
  c_syev  = T.c_THFloatLapack_syev
  c_geev  = T.c_THFloatLapack_geev
  c_gesvd = T.c_THFloatLapack_gesvd
  c_getrf = T.c_THFloatLapack_getrf
  c_getrs = T.c_THFloatLapack_getrs
  c_getri = T.c_THFloatLapack_getri
  p_gesv  = T.p_THFloatLapack_gesv
  p_trtrs = T.p_THFloatLapack_trtrs
  p_gels  = T.p_THFloatLapack_gels
  p_syev  = T.p_THFloatLapack_syev
  p_geev  = T.p_THFloatLapack_geev
  p_gesvd = T.p_THFloatLapack_gesvd
  p_getrf = T.p_THFloatLapack_getrf
  p_getrs = T.p_THFloatLapack_getrs
  p_getri = T.p_THFloatLapack_getri

instance THLapack THDoubleLapack where
  c_gesv  = T.c_THDoubleLapack_gesv
  c_trtrs = T.c_THDoubleLapack_trtrs
  c_gels  = T.c_THDoubleLapack_gels
  c_syev  = T.c_THDoubleLapack_syev
  c_geev  = T.c_THDoubleLapack_geev
  c_gesvd = T.c_THDoubleLapack_gesvd
  c_getrf = T.c_THDoubleLapack_getrf
  c_getrs = T.c_THDoubleLapack_getrs
  c_getri = T.c_THDoubleLapack_getri
  p_gesv  = T.p_THDoubleLapack_gesv
  p_trtrs = T.p_THDoubleLapack_trtrs
  p_gels  = T.p_THDoubleLapack_gels
  p_syev  = T.p_THDoubleLapack_syev
  p_geev  = T.p_THDoubleLapack_geev
  p_gesvd = T.p_THDoubleLapack_gesvd
  p_getrf = T.p_THDoubleLapack_getrf
  p_getrs = T.p_THDoubleLapack_getrs
  p_getri = T.p_THDoubleLapack_getri
