-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Float
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Reexports of Float-specific code from hasktorch-indef.
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Float (module X) where

import Numeric.Dimensions
import Torch.Types.TH as X

import Torch.Float.Types as X hiding (storage)
import Torch.Float.Index as X hiding (withDynamicState)
import Torch.Float.Mask  as X

import Torch.Indef.Float.Tensor as X
import Torch.Indef.Float.Tensor.Copy as X
import Torch.Indef.Float.Tensor.Index as X
import Torch.Indef.Float.Tensor.Masked as X
import Torch.Indef.Float.Tensor.Math as X
import Torch.Indef.Float.Tensor.Math.Compare as X
import Torch.Indef.Float.Tensor.Math.CompareT as X
import Torch.Indef.Float.Tensor.Math.Pairwise as X
import Torch.Indef.Float.Tensor.Math.Pointwise as X
import Torch.Indef.Float.Tensor.Math.Reduce as X
import Torch.Indef.Float.Tensor.Math.Scan as X
import Torch.Indef.Float.Tensor.Mode as X
import Torch.Indef.Float.Tensor.ScatterGather as X
import Torch.Indef.Float.Tensor.Sort as X
import Torch.Indef.Float.Tensor.TopK as X

import Torch.Indef.Float.Tensor.Math.Pointwise.Signed as X

import Torch.Indef.Float.NN as X
import Torch.Indef.Float.Tensor.Math.Blas as X
import Torch.Indef.Float.Tensor.Math.Floating as X
import Torch.Indef.Float.Tensor.Math.Lapack as X
import Torch.Indef.Float.Tensor.Math.Pointwise.Floating as X
import Torch.Indef.Float.Tensor.Math.Reduce.Floating as X

import Torch.Indef.Float.Tensor.Math.Random.TH as X
import Torch.Indef.Float.Tensor.Random.TH as X
import Torch.Core.Random as X (newRNG, seed, manualSeed, initialSeed)

-------------------------------------------------------------------------------

import System.IO.Unsafe

instance Dimensions d => Fractional (Tensor d) where
  fromRational = constant . fromRational
  (/) = (^/^)

instance Dimensions d => Floating (Tensor d) where
  pi = X.constant pi
  exp = X.exp
  log = X.log
  sqrt = X.sqrt
  sin = X.sin
  cos = X.cos
  asin = X.asin
  acos = X.acos
  atan = X.atan
  sinh = X.sinh
  cosh = X.cosh


