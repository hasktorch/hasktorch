module Torch.Indef.Static.Tensor.Math.Reduce where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Reduce        as Dynamic
import qualified Torch.Class.Tensor.Math.Reduce.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Reduce ()
import Torch.Indef.Dynamic.Tensor ()
import Torch.Indef.Static.Tensor ()

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

  _max :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  _max (r, ix) t1 = Dynamic._max (asDynamic r, longAsDynamic ix) (asDynamic t1)

  _min :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  _min (r, ix) t1 = Dynamic._min (asDynamic r, longAsDynamic ix) (asDynamic t1)

  _median :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  _median (r, ix) t1 = Dynamic._median (asDynamic r, longAsDynamic ix) (asDynamic t1)

  _sum r t = Dynamic._sum (asDynamic r) (asDynamic t)

  _prod :: Tensor d -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
  _prod r t = Dynamic._prod (asDynamic r) (asDynamic t)


