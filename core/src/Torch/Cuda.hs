-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Cuda
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Re-exports of all static CUDA-based tensors
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
module Torch.Cuda (module X) where

import Torch.Dimensions as X
import Torch.Types.THC as X
import Torch.Cuda.Storage as X

import Torch.Cuda.Byte.Static as X
import Torch.Cuda.Char.Static as X

import Torch.Cuda.Short.Static as X
import Torch.Cuda.Int.Static   as X
import Torch.Cuda.Long.Static  as X

import Torch.Cuda.Double.Static       as X
import Torch.Cuda.Double.StaticRandom as X

import qualified Numeric.Dimensions.Dim as D

-------------------------------------------------------------------------------

import System.IO.Unsafe

-- FIXME: these are... the same constraints...?)
instance (Dimensions d, D.Dimensions d) => Fractional (DoubleTensor (d::[Nat])) where
  fromRational = unsafePerformIO . constant . fromRational
  {-# NOINLINE fromRational #-}

  (/) = (^/^)
  {-# NOINLINE (/) #-}

instance (Dimensions d, D.Dimensions d) => Floating (DoubleTensor d) where
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

