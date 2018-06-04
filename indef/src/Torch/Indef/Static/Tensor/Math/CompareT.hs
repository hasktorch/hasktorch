module Torch.Indef.Static.Tensor.Math.CompareT
  ( ltTensor, ltTensorT, ltTensorT_
  , leTensor, leTensorT, leTensorT_
  , gtTensor, gtTensorT, gtTensorT_
  , geTensor, geTensorT, geTensorT_
  , neTensor, neTensorT, neTensorT_
  , eqTensor, eqTensorT, eqTensorT_
  ) where

import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.CompareT as Dynamic

ltTensor, leTensor, gtTensor, geTensor, neTensor, eqTensor
  :: (Dimensions d)
  => Tensor d -> Tensor d -> MaskTensor d
ltTensor a b = byteAsStatic $ Dynamic.ltTensor (asDynamic a) (asDynamic b)
leTensor a b = byteAsStatic $ Dynamic.leTensor (asDynamic a) (asDynamic b)
gtTensor a b = byteAsStatic $ Dynamic.gtTensor (asDynamic a) (asDynamic b)
geTensor a b = byteAsStatic $ Dynamic.geTensor (asDynamic a) (asDynamic b)
neTensor a b = byteAsStatic $ Dynamic.neTensor (asDynamic a) (asDynamic b)
eqTensor a b = byteAsStatic $ Dynamic.eqTensor (asDynamic a) (asDynamic b)

ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: Dimensions d => Tensor d -> Tensor d -> Tensor d
ltTensorT a b = asStatic $ Dynamic.ltTensorT (asDynamic a) (asDynamic b)
leTensorT a b = asStatic $ Dynamic.leTensorT (asDynamic a) (asDynamic b)
gtTensorT a b = asStatic $ Dynamic.gtTensorT (asDynamic a) (asDynamic b)
geTensorT a b = asStatic $ Dynamic.geTensorT (asDynamic a) (asDynamic b)
neTensorT a b = asStatic $ Dynamic.neTensorT (asDynamic a) (asDynamic b)
eqTensorT a b = asStatic $ Dynamic.eqTensorT (asDynamic a) (asDynamic b)

ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: Tensor d -> Tensor d -> IO (Tensor d)
ltTensorT_ a b = asStatic <$> Dynamic.ltTensorT_ (asDynamic a) (asDynamic b)
leTensorT_ a b = asStatic <$> Dynamic.leTensorT_ (asDynamic a) (asDynamic b)
gtTensorT_ a b = asStatic <$> Dynamic.gtTensorT_ (asDynamic a) (asDynamic b)
geTensorT_ a b = asStatic <$> Dynamic.geTensorT_ (asDynamic a) (asDynamic b)
neTensorT_ a b = asStatic <$> Dynamic.neTensorT_ (asDynamic a) (asDynamic b)
eqTensorT_ a b = asStatic <$> Dynamic.eqTensorT_ (asDynamic a) (asDynamic b)

