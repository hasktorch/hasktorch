module Torch.Indef.Static (module X) where

import Torch.Class.Types as X
  ( IsStatic(..)
  , Stride(..)
  , Size(..)
  , StorageOffset(..)
  , Step(..)
  , KeepDim(..), fromKeepDim, keep, ignore
  , SortOrder(..)
  )

import Torch.Class.Tensor.Static as X
import Torch.Indef.Static.Tensor as X

import Torch.Class.Tensor.Copy.Static as X
import Torch.Indef.Static.Tensor.Copy as X

import Torch.Class.Tensor.Index.Static as X
import Torch.Indef.Static.Tensor.Index as X

import Torch.Class.Tensor.Masked.Static as X
import Torch.Indef.Static.Tensor.Masked as X

import Torch.Class.Tensor.Math.Static as X
import Torch.Indef.Static.Tensor.Math as X

-- import Torch.Class.Tensor.Math.Blas as X
-- import Torch.Indef.Static.Tensor.Math.Blas as X

import Torch.Class.Tensor.Math.Compare.Static as X
import Torch.Indef.Static.Tensor.Math.Compare as X

import Torch.Class.Tensor.Math.CompareT.Static as X
import Torch.Indef.Static.Tensor.Math.CompareT as X

import Torch.Class.Tensor.Math.Pairwise.Static as X
import Torch.Indef.Static.Tensor.Math.Pairwise as X

import Torch.Class.Tensor.Math.Pointwise.Static as X
import Torch.Indef.Static.Tensor.Math.Pointwise as X

-- import Torch.Class.Tensor.Math.Random as X
-- import Torch.Indef.Static.Tensor.Math.Random as X

import Torch.Class.Tensor.Math.Reduce.Static as X
import Torch.Indef.Static.Tensor.Math.Reduce as X

import Torch.Class.Tensor.Math.Scan.Static as X
import Torch.Indef.Static.Tensor.Math.Scan as X

import Torch.Class.Tensor.Mode.Static as X
import Torch.Indef.Static.Tensor.Mode as X

import Torch.Class.Tensor.ScatterGather.Static as X
import Torch.Indef.Static.Tensor.ScatterGather as X

import Torch.Class.Tensor.Sort.Static as X
import Torch.Indef.Static.Tensor.Sort as X

import Torch.Class.Tensor.TopK.Static as X
import Torch.Indef.Static.Tensor.TopK as X

-- import Torch.Class.Tensor.Random as X
-- import Torch.Indef.Static.Tensor.Random as X

