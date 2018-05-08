{-# OPTIONS_GHC -fno-cse #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Float (module X) where

import Torch.Types.TH as X
import Torch.Indef.Float.Types as X hiding (storage)
import Torch.Indef.Float.Index as X

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
  fromRational = unsafePerformIO . constant . fromRational
  {-# NOINLINE fromRational #-}

  (/) = (^/^)
  {-# NOINLINE (/) #-}

instance Dimensions d => Floating (Tensor d) where
  pi = unsafePerformIO $ X.constant pi
  {-# NOINLINE pi #-}

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


