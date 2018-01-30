module Torch.Raw.Lapack
  ( THLapack(..)
  , module X
  ) where

import Torch.Raw.Internal as X

class THLapack t where
  c_gesv :: CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_gels :: CChar -> CInt -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_syev :: CChar -> CChar -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_geev :: CChar -> CChar -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_getrf :: CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> CIntPtr -> IO ()
  c_getrs :: CChar -> CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ()
  c_getri :: CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ()
  p_gesv :: FunPtr (CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_trtrs :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_gels :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_syev :: FunPtr (CChar -> CChar -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_geev :: FunPtr (CChar -> CChar -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_gesvd :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr t -> CInt -> Ptr t -> Ptr t -> CInt -> Ptr t -> CInt -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_getrf :: FunPtr (CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> CIntPtr -> IO ())
  p_getrs :: FunPtr (CChar -> CInt -> CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ())
  p_getri :: FunPtr (CInt -> Ptr t -> CInt -> CIntPtr -> Ptr t -> CInt -> CIntPtr -> IO ())
