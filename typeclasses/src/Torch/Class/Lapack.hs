module Torch.Class.Lapack where

import Foreign.C.Types (CIntPtr)

class Lapack t where
  gesv  :: Int -> Int -> t -> Int -> CIntPtr -> t -> Int -> CIntPtr -> IO ()
  trtrs :: Word -> Word -> Word -> Int -> Int -> t -> Int -> t -> Int -> CIntPtr -> IO ()
  gels  :: Word -> Int -> Int -> Int -> t -> Int -> t -> Int -> t -> Int -> CIntPtr -> IO ()
  syev  :: Word -> Word -> Int -> t -> Int -> t -> t -> Int -> CIntPtr -> IO ()
  geev  :: Word -> Word -> Int -> t -> Int -> t -> t -> t -> Int -> t -> Int -> t -> Int -> CIntPtr -> IO ()
  gesvd :: Word -> Word -> Int -> Int -> t -> Int -> t -> t -> Int -> t -> Int -> t -> Int -> CIntPtr -> IO ()
  getrf :: Int -> Int -> t -> Int -> CIntPtr -> CIntPtr -> IO ()
  getrs :: Word -> Int -> Int -> t -> Int -> CIntPtr -> t -> Int -> CIntPtr -> IO ()
  getri :: Int -> t -> Int -> CIntPtr -> t -> Int -> CIntPtr -> IO ()
