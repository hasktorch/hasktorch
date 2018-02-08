module Torch.Class.C.Blas where

import THTypes

class Blas t where
  swap :: Integer -> t -> Integer -> t -> Integer -> IO ()
  scal :: Integer -> t -> t -> Integer -> IO ()
  copy :: Integer -> t -> Integer -> t -> Integer -> IO ()
  axpy :: Integer -> t -> t -> Integer -> t -> Integer -> IO ()
  dot  :: Integer -> t -> Integer -> t -> Integer -> t
  gemv :: Word -> Integer -> Integer -> t -> t -> Integer -> t -> Integer -> t -> t -> Integer -> IO ()
  ger  :: Integer -> Integer -> t -> t -> Integer -> t -> Integer -> t-> Integer -> IO ()
