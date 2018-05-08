module Torch.Indef.Static.Tensor.Math.Pointwise.Signed where

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math (constant)
import Torch.Indef.Static.Tensor.Math.Pointwise (sign, (^*^), (^-^), (^+^))
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed as Dynamic

import System.IO.Unsafe

_abs :: Tensor d -> Tensor d -> IO ()
_abs r t = Dynamic._abs (asDynamic r) (asDynamic t)

_neg :: Tensor d -> Tensor d -> IO ()
_neg r t = Dynamic._neg (asDynamic r) (asDynamic t)

neg, abs :: (Dimensions d) => Tensor d -> IO (Tensor d)
neg t = withEmpty (`_neg` t)
abs t = withEmpty (`_abs` t)


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

  fromInteger = unsafePerformIO . constant . fromIntegral
  {-# NOINLINE fromInteger #-}
