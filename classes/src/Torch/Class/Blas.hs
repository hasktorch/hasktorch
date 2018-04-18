module Torch.Class.Blas where

class Blas rs r where
  swap :: Integer -> rs -> Integer -> rs -> Integer -> IO ()
  scal :: Integer -> r -> rs -> Integer -> IO ()
  copy :: Integer -> rs -> Integer -> rs -> Integer -> IO ()
  axpy :: Integer -> r -> rs -> Integer -> rs -> Integer -> IO ()
  dot  :: Integer -> rs -> Integer -> rs -> Integer -> r
  gemv :: Int -> Integer -> Integer -> r -> rs -> Integer -> rs -> Integer -> r -> rs -> Integer -> IO ()
  ger  :: Integer -> Integer -> r -> rs -> Integer -> rs -> Integer -> rs -> Integer -> IO ()

