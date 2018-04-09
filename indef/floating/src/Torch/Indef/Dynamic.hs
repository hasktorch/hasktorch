{-# LANGUAGE PackageImports #-}
{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -fno-cse  #-}
module Torch.Indef.Dynamic (module X) where

import "hasktorch-indef-signed" Torch.Indef.Dynamic as X

-- Typeclasses already included
import Torch.Indef.Dynamic.Tensor.Math.Floating as X
import Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating as X
import Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating as X

-- new typeclasses
import Torch.Class.Tensor.Math.Blas as X
import Torch.Indef.Dynamic.Tensor.Math.Blas as X

import Torch.Class.Tensor.Math.Lapack as X
import Torch.Indef.Dynamic.Tensor.Math.Lapack as X

import Torch.Class.NN as X
import Torch.Indef.Dynamic.NN as X

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



