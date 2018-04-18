{-# LANGUAGE PackageImports #-}
module Torch.Indef.Static (module X) where

import "hasktorch-indef-signed" Torch.Indef.Static as X

-- Typeclasses already included
import Torch.Indef.Static.Tensor.Math.Floating as X
import Torch.Indef.Static.Tensor.Math.Reduce.Floating as X
import Torch.Indef.Static.Tensor.Math.Pointwise.Floating as X

-- new typeclasses
import Torch.Class.Tensor.Math.Blas.Static as X
import Torch.Indef.Static.Tensor.Math.Blas as X

import Torch.Class.Tensor.Math.Lapack.Static as X
import Torch.Indef.Static.Tensor.Math.Lapack as X

import Torch.Class.NN.Static as X
import Torch.Indef.Static.NN as X


