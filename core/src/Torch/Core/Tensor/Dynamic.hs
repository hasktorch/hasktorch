{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Core.Tensor.Dynamic
  ( ByteTensor
  , ShortTensor
  , IntTensor
  , LongTensor
  , FloatTensor
  , DoubleTensor

  , module X
  , module Classes
  , LapackClass.TensorLapack(..)
  ) where

import THTypes
import Foreign (withForeignPtr)
import GHC.Int
import Torch.Class.C.Internal
import Torch.Core.Tensor.Dim
import qualified Torch.Class.C.Tensor as C
import Torch.Class.C.Storage (IsStorage)
import qualified Torch.Class.C.Storage as Storage
import qualified Torch.Class.C.Tensor.Lapack as LapackClass

import Torch.Core.Tensor.Dynamic.Copy as Classes
import Torch.Core.Tensor.Dynamic.Conv as Classes
import Torch.Core.Tensor.Dynamic.Math as Classes
import Torch.Core.Tensor.Dynamic.Random as Classes

import qualified Torch.Core.ByteTensor.Dynamic as B
import qualified Torch.Core.ShortTensor.Dynamic as S
import qualified Torch.Core.IntTensor.Dynamic as I
import qualified Torch.Core.LongTensor.Dynamic as L
import qualified Torch.Core.FloatTensor.Dynamic as F
import qualified Torch.Core.DoubleTensor.Dynamic as D

import Torch.Class.C.IsTensor as X
import Torch.Core.ByteTensor.Dynamic.IsTensor ()
import Torch.Core.ShortTensor.Dynamic.IsTensor ()
import Torch.Core.IntTensor.Dynamic.IsTensor ()
import Torch.Core.LongTensor.Dynamic.IsTensor ()
import Torch.Core.FloatTensor.Dynamic.IsTensor ()
import Torch.Core.DoubleTensor.Dynamic.IsTensor ()

import Torch.Core.FloatTensor.Dynamic.Lapack ()
import Torch.Core.DoubleTensor.Dynamic.Lapack ()

type ByteTensor = B.Tensor
-- type CharTensor = C.Tensor
type ShortTensor = S.Tensor
type IntTensor = I.Tensor
type LongTensor = L.Tensor
-- type HalfTensor = H.Tensor
type FloatTensor = F.Tensor
type DoubleTensor = D.Tensor


