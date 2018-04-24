{-# OPTIONS_GHC -fno-cse  #-}
module Torch.Indef.Dynamic (module X) where

import System.IO.Unsafe (unsafePerformIO)
import Torch.Dimensions
import Torch.Indef.Types

-------------------------------------------------------------------------------

import Torch.Class.Types as X
  ( Stride(..)
  , Size(..)
  , StorageOffset(..)
  , Step(..)
  , KeepDim(..), fromKeepDim, keep, ignore
  , SortOrder(..)
  )

import Torch.Class.Tensor as X
import Torch.Indef.Dynamic.Tensor as X

import Torch.Class.Tensor.Copy as X
import Torch.Indef.Dynamic.Tensor.Copy as X

import Torch.Class.Tensor.Index as X
import Torch.Indef.Dynamic.Tensor.Index as X

import Torch.Class.Tensor.Masked as X
import Torch.Indef.Dynamic.Tensor.Masked as X

import Torch.Class.Tensor.Math as X
import Torch.Indef.Dynamic.Tensor.Math as X

import Torch.Class.Tensor.Math.Compare as X
import Torch.Indef.Dynamic.Tensor.Math.Compare as X

import Torch.Class.Tensor.Math.CompareT as X
import Torch.Indef.Dynamic.Tensor.Math.CompareT as X

import Torch.Class.Tensor.Math.Pairwise as X
import Torch.Indef.Dynamic.Tensor.Math.Pairwise as X

import Torch.Class.Tensor.Math.Pointwise as X
import Torch.Indef.Dynamic.Tensor.Math.Pointwise as X

import Torch.Class.Tensor.Math.Reduce as X
import Torch.Indef.Dynamic.Tensor.Math.Reduce as X

import Torch.Class.Tensor.Math.Scan as X
import Torch.Indef.Dynamic.Tensor.Math.Scan as X

import Torch.Class.Tensor.Mode as X
import Torch.Indef.Dynamic.Tensor.Mode as X

import Torch.Class.Tensor.ScatterGather as X
import Torch.Indef.Dynamic.Tensor.ScatterGather as X

import Torch.Class.Tensor.Sort as X
import Torch.Indef.Dynamic.Tensor.Sort as X

import Torch.Class.Tensor.TopK as X
import Torch.Indef.Dynamic.Tensor.TopK as X

instance Show Dynamic where
  show t = unsafePerformIO $ do
    ds <- (fmap.fmap) fromIntegral (getDimList t)
    (vs, desc) <- showTensor (get1d t) (get2d t) (get3d t) (get4d t) ds
    pure (vs ++ "\n" ++ desc)
  {-# NOINLINE show #-}

