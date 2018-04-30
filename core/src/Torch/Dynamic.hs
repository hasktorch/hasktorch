-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Cuda.Dynamic
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Re-exports of all CUDA-based dynamic tensor code
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -fno-cse  #-}
module Torch.Dynamic (module X) where

import Torch.Dimensions as X
import Torch.Types.TH   as X
  hiding ( HalfTensor, FloatTensor, DoubleTensor, ShortTensor
         , IntTensor, LongTensor, CharTensor, ByteTensor)

import Torch.Byte.Dynamic as X
import Torch.Char.Dynamic as X

import Torch.Short.Dynamic as X
import Torch.Int.Dynamic   as X
import Torch.Long.Dynamic  as X

import Torch.Float.Dynamic  as X
import Torch.Float.DynamicRandom  as X
import Torch.Double.Dynamic as X
import Torch.Double.DynamicRandom as X

-------------------------------------------------------------------------------

import System.IO.Unsafe

instance Fractional DoubleDynamic where
  fromRational = unsafePerformIO . X.constant (dim :: Dim '[1]) . fromRational
  {-# NOINLINE fromRational #-}

  (/) = (^/^)
  {-# NOINLINE (/) #-}

instance Fractional FloatDynamic where
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

instance Floating FloatDynamic where
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

