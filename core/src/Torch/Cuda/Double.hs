-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Cuda.Double
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Cuda.Double (module X) where

import Numeric.Dimensions
import Torch.Types.THC as X
import Torch.Cuda.Double.NN as X

import Torch.Cuda.Double.Types as X hiding (storage)
import Torch.Cuda.Double.Index as X hiding (withDynamicState)
import Torch.Cuda.Double.Mask  as X

import Torch.Indef.Cuda.Double.Tensor as X
import Torch.Indef.Cuda.Double.Tensor.Copy as X
import Torch.Indef.Cuda.Double.Tensor.Index as X
import Torch.Indef.Cuda.Double.Tensor.Masked as X
import Torch.Indef.Cuda.Double.Tensor.Math as X
import Torch.Indef.Cuda.Double.Tensor.Math.Compare as X
import Torch.Indef.Cuda.Double.Tensor.Math.CompareT as X
import Torch.Indef.Cuda.Double.Tensor.Math.Pairwise as X
import Torch.Indef.Cuda.Double.Tensor.Math.Pointwise as X
import Torch.Indef.Cuda.Double.Tensor.Math.Reduce as X
import Torch.Indef.Cuda.Double.Tensor.Math.Scan as X
import Torch.Indef.Cuda.Double.Tensor.Mode as X
import Torch.Indef.Cuda.Double.Tensor.ScatterGather as X
import Torch.Indef.Cuda.Double.Tensor.Sort as X
import Torch.Indef.Cuda.Double.Tensor.TopK as X

import Torch.Indef.Cuda.Double.Tensor.Math.Pointwise.Signed as X

import Torch.Indef.Cuda.Double.Tensor.Math.Blas as X
import Torch.Indef.Cuda.Double.Tensor.Math.Floating as X
import Torch.Indef.Cuda.Double.Tensor.Math.Lapack as X
import Torch.Indef.Cuda.Double.Tensor.Math.Pointwise.Floating as X
import Torch.Indef.Cuda.Double.Tensor.Math.Reduce.Floating as X

import Torch.Indef.Cuda.Double.Tensor.Random.THC as X

-------------------------------------------------------------------------------

import System.IO.Unsafe

instance Dimensions d => Fractional (Tensor d) where
  fromRational = constant . fromRational

  (/) = (^/^)
  {-# NOINLINE (/) #-}

instance Dimensions d => Floating (Tensor d) where
  pi = X.constant pi

  exp = unsafePerformIO . X.exp
  {-# NOINLINE exp #-}

  log = unsafePerformIO . X.log
  {-# NOINLINE log #-}

  sqrt = unsafePerformIO . X.sqrt
  {-# NOINLINE sqrt #-}

  sin = unsafePerformIO . X.sin
  {-# NOINLINE sin #-}

  cos = unsafePerformIO . X.cos
  {-# NOINLINE cos #-}

  asin = unsafePerformIO . X.asin
  {-# NOINLINE asin #-}

  acos = unsafePerformIO . X.acos
  {-# NOINLINE acos #-}

  atan = unsafePerformIO . X.atan
  {-# NOINLINE atan #-}

  sinh = unsafePerformIO . X.sinh
  {-# NOINLINE sinh #-}

  cosh = unsafePerformIO . X.cosh
  {-# NOINLINE cosh #-}


