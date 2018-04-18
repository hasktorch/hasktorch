{-# LANGUAGE PackageImports #-}
{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -fno-cse  #-}
module Torch.Indef.Static (module X) where

import "hasktorch-indef-unsigned" Torch.Indef.Static as X
import Torch.Indef.Static.Tensor.Math.Pointwise.Signed as X

-------------------------------------------------------------------------------

import System.IO.Unsafe
import Torch.Dimensions
import qualified Torch.Sig.Types as Sig (Tensor)

instance Dimensions d => Num (Sig.Tensor d) where
  (+) = (^+^)
  {-# NOINLINE (+) #-}

  (-) = (^-^)
  {-# NOINLINE (-) #-}

  (*) = (^*^)
  {-# NOINLINE (*) #-}

  abs = unsafePerformIO . X.abs
  {-# NOINLINE abs #-}

  signum = unsafePerformIO . X.sign
  {-# NOINLINE signum #-}

  fromInteger = unsafePerformIO . X.constant . fromIntegral
  {-# NOINLINE fromInteger #-}

