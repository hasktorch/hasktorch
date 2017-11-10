module HMatrixBridge where

import Data.Maybe (fromJust)
import Numeric.LinearAlgebra hiding (size, disp)

import THDoubleTensor
import THDoubleTensorMath

import Foreign
import Foreign.C.Types

-- TODO: conversion to/from hmatrix types
