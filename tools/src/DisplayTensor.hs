{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

-- Pretty printing of low level tensor representations
-- approximately
-- https://github.com/pytorch/pytorch/blob/49ec984c406e67107aae2891d24c8839b7dc7c33/torch/_tensor_str.py

module Main where

import Data.Maybe (fromJust)

import Foreign
import Foreign.C.Types
import THTypes
import TorchTensor

import THDoubleTensor
import THDoubleTensorMath
import TorchTensor
import THDoubleTensorRandom

-- tensorStr tensor = undefined
--   where
--     n = edgeitems defaultPrintOptions
--     sz = size tensor
--     has_hdots = last sz > 2 * n
--     has_vdots = (head . drop 1 . reverse $ sz) > 2 * n
--     print_full_mat = not has_hdots && not has_vdots
--     -- what to do for formatter ?
--     -- print_dots = product >= threshold defaultPrintOptions


