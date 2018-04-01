module Torch.Indef.Static.Tensor.Math.Reduce where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Reduce        as Dynamic
import qualified Torch.Class.Tensor.Math.Reduce.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Reduce ()

instance Class.TensorMathReduce Tensor where
  minall :: Tensor d -> IO HsReal
  minall t = Dynamic.minall (asDynamic t)

  maxall :: Tensor d -> IO HsReal
  maxall t = Dynamic.maxall (asDynamic t)

  medianall :: Tensor d -> IO HsReal
  medianall t = Dynamic.medianall (asDynamic t)

  sumall :: Tensor d -> IO HsAccReal
  sumall t = Dynamic.sumall (asDynamic t)

  prodall :: Tensor d -> IO HsAccReal
  prodall t = Dynamic.prodall (asDynamic t)

  max_ :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  max_ (r, ix) t1 = Dynamic.max_ (asDynamic r, longAsDynamic ix) (asDynamic t1)

  min_ :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  min_ (r, ix) t1 = Dynamic.min_ (asDynamic r, longAsDynamic ix) (asDynamic t1)

  median_ :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  median_ (r, ix) t1 = Dynamic.median_ (asDynamic r, longAsDynamic ix) (asDynamic t1)

  sum_ r t = Dynamic.sum_ (asDynamic r) (asDynamic t)

  prod_ :: Tensor d -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  prod_ r t = Dynamic.prod_ (asDynamic r) (asDynamic t)


