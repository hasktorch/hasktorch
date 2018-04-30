-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Cuda.Dynamic
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Re-exports of all dynamic cuda-based tensors
-------------------------------------------------------------------------------


module Torch.Cuda.Dynamic (module X) where

import Torch.Cuda.Byte.Dynamic as X
import Torch.Cuda.Char.Dynamic as X

import Torch.Cuda.Short.Dynamic as X
import Torch.Cuda.Int.Dynamic   as X
import Torch.Cuda.Long.Dynamic  as X

import Torch.Cuda.Double.Dynamic       as X
import Torch.Cuda.Double.DynamicRandom as X

import Torch.Dimensions as X
import Torch.Types.THC as X

-------------------------------------------------------------------------------

import System.IO.Unsafe

instance Fractional DoubleDynamic where
  fromRational = unsafePerformIO . X.constant (dim :: Dim '[1]) . fromRational
  {-# NOINLINE fromRational #-}

  (/) = (^/^)
  {-# NOINLINE (/) #-}

instance Floating DoubleDynamic where
  pi = unsafePerformIO $ X.constant (dim :: Dim '[1]) pi
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

