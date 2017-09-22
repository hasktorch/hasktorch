{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

-- Pretty printing of low level tensor representations
-- approximately
-- https://github.com/pytorch/pytorch/blob/49ec984c406e67107aae2891d24c8839b7dc7c33/torch/_tensor_str.py

module Main where

import Foreign
import Foreign.C.Types
import THTypes
import TorchTensor

import THDoubleTensor
import THDoubleTensorMath
import TorchTensor
-- import THDoubleTensorRandom

data PrintOptions = PrintOptions {
  precision :: Int,
  threshold :: Int,
  edgeitems :: Int,
  linewidth :: Int
  }

defaultPrintOptions = PrintOptions {
  precision = 4,
  threshold = 1000,
  edgeitems = 3,
  linewidth = 80
  }

tensorStr tensor = undefined
  where
    n = edgeitems defaultPrintOptions
    sz = size tensor
    has_hdots = last sz > 2 * n
    has_vdots = (head . drop 1 . reverse $ sz) > 2 * n
    print_full_mat = not has_hdots && not has_vdots
    -- what to do for formatter ?
    -- print_dots = product >= threshold defaultPrintOptions

main = do
  putStrLn "Done"

  -- t1 <- c_THDoubleTensor_newWithSize2d 2 2
  -- t2 <- c_THDoubleTensor_newWithSize2d 2 2
  -- t3 <- c_THDoubleTensor_newWithSize2d 2 2
  -- c_THDoubleTensor_fill t1 3.0
  -- print $ c_THDoubleTensor_get2d t1 0 0
  -- c_THDoubleTensor_fill t2 4.0
  -- print $ c_THDoubleTensor_get2d t2 0 0
  -- print $ c_THDoubleTensor_dot t1 t2
  -- c_THDoubleTensor_free t1
  -- c_THDoubleTensor_free t2
  -- c_THDoubleTensor_free t3


