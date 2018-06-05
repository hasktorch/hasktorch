module Torch.Indef.Static.Tensor.Math.Reduce.Floating where

import Torch.Dimensions

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating as Dynamic

dist    :: Tensor d -> Tensor d' -> HsReal -> IO (HsAccReal)
dist r t = Dynamic.dist (asDynamic r) (asDynamic t)

_var :: Tensor d -> Tensor d' -> Int -> Int -> Int -> IO ()
_var r t = Dynamic._var (asDynamic r) (asDynamic t)

varall  :: Tensor d -> Int -> IO (HsAccReal)
varall t = Dynamic.varall (asDynamic t)

_std     :: Tensor d -> Tensor d' -> Int -> Int -> Int -> IO ()
_std r t = Dynamic._std (asDynamic r) (asDynamic t)

stdall  :: Tensor d -> Int -> IO (HsAccReal)
stdall t = Dynamic.stdall (asDynamic t)

_renorm  :: Tensor d -> Tensor d' -> HsReal -> Int -> HsReal -> IO ()
_renorm r t = Dynamic._renorm (asDynamic r) (asDynamic t)

_norm    :: Tensor d -> Tensor d' -> HsReal -> Int -> Int -> IO ()
_norm r t = Dynamic._norm (asDynamic r) (asDynamic t)

normall :: Tensor d -> HsReal -> IO HsAccReal
normall t = Dynamic.normall (asDynamic t)

_mean    :: Tensor d -> Tensor d' -> Int -> Int -> IO ()
_mean r t = Dynamic._mean (asDynamic r) (asDynamic t)

meanall :: Tensor d -> IO HsAccReal
meanall t = Dynamic.meanall (asDynamic t)


