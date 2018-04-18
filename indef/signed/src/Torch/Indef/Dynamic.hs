{-# LANGUAGE PackageImports #-}
{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -fno-cse  #-}
module Torch.Indef.Dynamic (module X) where

import "hasktorch-indef-unsigned" Torch.Indef.Dynamic as X
import Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed as X

-------------------------------------------------------------------------------

import System.IO.Unsafe
import Torch.Sig.Types (Dynamic)
import Torch.Dimensions

instance Num Dynamic where
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

  fromInteger = unsafePerformIO . X.constant (dim :: Dim '[1]) . fromIntegral
  {-# NOINLINE fromInteger #-}

