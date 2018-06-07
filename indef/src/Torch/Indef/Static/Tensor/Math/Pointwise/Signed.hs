-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Pointwise.Signed
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Math.Pointwise.Signed where

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math (constant)
import Torch.Indef.Static.Tensor.Math.Pointwise (sign, (^*^), (^-^), (^+^))
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed as Dynamic

import Numeric.Dimensions
import System.IO.Unsafe

-- | Static call to 'Dynamic.abs_'
abs_ :: Tensor d -> IO ()
abs_ t = Dynamic.abs_ (asDynamic t)

-- | Static call to 'Dynamic.neg_'
neg_ :: Tensor d -> IO ()
neg_ t = Dynamic.neg_ (asDynamic t)

-- | Static call to 'Dynamic.neg'
neg :: Dimensions d => Tensor d -> IO (Tensor d)
neg t = asStatic <$> Dynamic.neg (asDynamic t)

-- | Static call to 'Dynamic.abs'
abs :: Dimensions d => Tensor d -> IO (Tensor d)
abs t = asStatic <$> Dynamic.abs (asDynamic t)


instance Dimensions d => Num (Tensor d) where
  (+) = (^+^)
  {-# NOINLINE (+) #-}

  (-) = (^-^)
  {-# NOINLINE (-) #-}

  (*) = (^*^)
  {-# NOINLINE (*) #-}

  abs = unsafePerformIO . Torch.Indef.Static.Tensor.Math.Pointwise.Signed.abs
  {-# NOINLINE abs #-}

  signum = unsafePerformIO . sign
  {-# NOINLINE signum #-}

  fromInteger = constant . fromIntegral
