module Torch.Indef.Static.Tensor.Math.Reduce.Floating where

import GHC.Int
import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Reduce as Dynamic
import qualified Torch.Class.Tensor.Math.Reduce.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating ()

instance Class.TensorMathReduceFloating Tensor where
  dist    :: Tensor d -> Tensor d' -> HsReal -> IO (HsAccReal)
  dist r t = Dynamic.dist (asDynamic r) (asDynamic t)

  var_ :: Tensor d -> Tensor d' -> Int -> Int -> Int -> IO ()
  var_ r t = Dynamic.var_ (asDynamic r) (asDynamic t)

  varall  :: Tensor d -> Int -> IO (HsAccReal)
  varall t = Dynamic.varall (asDynamic t)

  std_     :: Tensor d -> Tensor d' -> Int -> Int -> Int -> IO ()
  std_ r t = Dynamic.std_ (asDynamic r) (asDynamic t)

  stdall  :: Tensor d -> Int -> IO (HsAccReal)
  stdall t = Dynamic.stdall (asDynamic t)

  renorm_  :: Tensor d -> Tensor d' -> HsReal -> Int -> HsReal -> IO ()
  renorm_ r t = Dynamic.renorm_ (asDynamic r) (asDynamic t)

  norm_    :: Tensor d -> Tensor d' -> HsReal -> Int -> Int -> IO ()
  norm_ r t = Dynamic.norm_ (asDynamic r) (asDynamic t)

  normall :: Tensor d -> HsReal -> IO (HsAccReal)
  normall t = Dynamic.normall (asDynamic t)

  mean_    :: Tensor d -> Tensor d' -> Int -> Int -> IO ()
  mean_ r t = Dynamic.mean_ (asDynamic r) (asDynamic t)

  meanall :: Tensor d -> IO (HsAccReal)
  meanall t = Dynamic.meanall (asDynamic t)


