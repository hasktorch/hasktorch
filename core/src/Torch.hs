-------------------------------------------------------------------------------
-- |
-- Module    :  Torch
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Re-exports of all static CPU-based tensors
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -fno-cse  #-}
module Torch ( module X ) where

import Torch.Dimensions as X
import Torch.Types.TH as X

import Torch.Byte.Static as X
import Torch.Char.Static as X

import Torch.Short.Static as X
import Torch.Int.Static   as X
import Torch.Long.Static  as X

import Torch.Float.Static  as X
import Torch.Float.StaticRandom  as X
import Torch.Double.Static as X
import Torch.Double.StaticRandom as X

-------------------------------------------------------------------------------

import System.IO.Unsafe

instance Dimensions d => Fractional (DoubleTensor d) where
  fromRational = unsafePerformIO . constant . fromRational
  {-# NOINLINE fromRational #-}

  (/) = (^/^)
  {-# NOINLINE (/) #-}

instance Dimensions d => Fractional (FloatTensor d) where
  fromRational = unsafePerformIO . constant . fromRational
  {-# NOINLINE fromRational #-}

  (/) = (^/^)
  {-# NOINLINE (/) #-}

instance Dimensions d => Floating (DoubleTensor d) where
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

instance Dimensions d => Floating (FloatTensor d) where
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

