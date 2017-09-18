{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

-- Pretty printing of low level tensor representations

module Main where

import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom

import Foreign
import Foreign.C.Types
import THTypes

size :: (Ptr CTHDoubleTensor) -> [CLong]
size t =
  fmap (\x -> c_THDoubleTensor_size t x) [0..maxdim]
  where
    maxdim = (c_THDoubleTensor_nDimension t) - 1

main = do
  t1 <- c_THDoubleTensor_newWithSize2d 2 2
  t2 <- c_THDoubleTensor_newWithSize2d 2 2
  t3 <- c_THDoubleTensor_newWithSize2d 2 2

  c_THDoubleTensor_fill t1 3.0
  print $ c_THDoubleTensor_get2d t1 0 0
  c_THDoubleTensor_fill t2 4.0
  print $ c_THDoubleTensor_get2d t2 0 0
  print $ c_THDoubleTensor_dot t1 t2
  c_THDoubleTensor_free t1
  c_THDoubleTensor_free t2
  c_THDoubleTensor_free t3


