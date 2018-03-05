module Torch.Class.Blas where

import Torch.Types.TH

class Blas real where
  swap :: Integer -> real -> Integer -> real -> Integer -> IO ()
  scal :: Integer -> real -> real -> Integer -> IO ()
  copy :: Integer -> real -> Integer -> real -> Integer -> IO ()
  axpy :: Integer -> real -> real -> Integer -> real -> Integer -> IO ()
  dot  :: Integer -> real -> Integer -> real -> Integer -> t
  gemv :: Word -> Integer -> Integer -> real -> real -> Integer -> real -> Integer -> real -> real -> Integer -> IO ()
  ger  :: Integer -> Integer -> real -> real -> Integer -> real -> Integer -> t-> Integer -> IO ()
