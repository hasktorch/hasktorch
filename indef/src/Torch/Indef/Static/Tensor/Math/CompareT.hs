-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.CompareT
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Math.CompareT
  ( ltTensor, ltTensorT, ltTensorT_
  , leTensor, leTensorT, leTensorT_
  , gtTensor, gtTensorT, gtTensorT_
  , geTensor, geTensorT, geTensorT_
  , neTensor, neTensorT, neTensorT_
  , eqTensor, eqTensorT, eqTensorT_
  ) where

import Numeric.Dimensions
import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.CompareT as Dynamic

-- | Return a byte tensor which contains boolean values indicating the relation between two tensors.
ltTensor, leTensor, gtTensor, geTensor, neTensor, eqTensor
  :: (Dimensions d)
  => Tensor d -> Tensor d -> MaskTensor d
ltTensor a b = byteAsStatic $ Dynamic.ltTensor (asDynamic a) (asDynamic b)
leTensor a b = byteAsStatic $ Dynamic.leTensor (asDynamic a) (asDynamic b)
gtTensor a b = byteAsStatic $ Dynamic.gtTensor (asDynamic a) (asDynamic b)
geTensor a b = byteAsStatic $ Dynamic.geTensor (asDynamic a) (asDynamic b)
neTensor a b = byteAsStatic $ Dynamic.neTensor (asDynamic a) (asDynamic b)
eqTensor a b = byteAsStatic $ Dynamic.eqTensor (asDynamic a) (asDynamic b)

-- | return a tensor which contains numeric values indicating the relation between two tensors.
-- 0 stands for false, 1 stands for true.
ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: Dimensions d
  => Tensor d  -- ^ source tensor.
  -> Tensor d  -- ^ tensor to compare with.
  -> Tensor d  -- ^ new return tensor.
ltTensorT a b = asStatic $ Dynamic.ltTensorT (asDynamic a) (asDynamic b)
leTensorT a b = asStatic $ Dynamic.leTensorT (asDynamic a) (asDynamic b)
gtTensorT a b = asStatic $ Dynamic.gtTensorT (asDynamic a) (asDynamic b)
geTensorT a b = asStatic $ Dynamic.geTensorT (asDynamic a) (asDynamic b)
neTensorT a b = asStatic $ Dynamic.neTensorT (asDynamic a) (asDynamic b)
eqTensorT a b = asStatic $ Dynamic.eqTensorT (asDynamic a) (asDynamic b)

-- | mutate a tensor in-place with its numeric relation to the second tensor of the same size,
-- where 0 stands for false and 1 stands for true.
ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: Tensor d  -- ^ source tensor to mutate inplace.
  -> Tensor d  -- ^ tensor to compare with.
  -> IO ()
ltTensorT_ a b = Dynamic.ltTensorT_ (asDynamic a) (asDynamic b)
leTensorT_ a b = Dynamic.leTensorT_ (asDynamic a) (asDynamic b)
gtTensorT_ a b = Dynamic.gtTensorT_ (asDynamic a) (asDynamic b)
geTensorT_ a b = Dynamic.geTensorT_ (asDynamic a) (asDynamic b)
neTensorT_ a b = Dynamic.neTensorT_ (asDynamic a) (asDynamic b)
eqTensorT_ a b = Dynamic.eqTensorT_ (asDynamic a) (asDynamic b)

