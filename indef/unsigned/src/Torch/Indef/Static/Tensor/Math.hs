module Torch.Indef.Static.Tensor.Math where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Static as Class
import qualified Torch.Class.Tensor.Math as Dynamic
import qualified Torch.Types.TH as TH

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math ()

instance Class.TensorMath Tensor where
  fill_ :: Tensor d -> HsReal -> IO ()
  fill_ r = Dynamic.fill_ (asDynamic r)

  zero_ :: Tensor d -> IO ()
  zero_ r = Dynamic.zero_ (asDynamic r)

  zeros_ :: Tensor d -> IndexStorage -> IO ()
  zeros_ r = Dynamic.zeros_ (asDynamic r)

  zerosLike_ :: Tensor d -> Tensor d -> IO ()
  zerosLike_ r t = Dynamic.zerosLike_ (asDynamic r) (asDynamic t)
  ones_ :: Tensor d -> TH.IndexStorage -> IO ()
  ones_ r = Dynamic.ones_ (asDynamic r)
  onesLike_ :: Tensor d -> Tensor d -> IO ()
  onesLike_ r t = Dynamic.onesLike_ (asDynamic r) (asDynamic t)
  numel :: Tensor d -> IO Integer
  numel t = Dynamic.numel (asDynamic t)
  reshape_ :: Tensor d -> Tensor d -> TH.IndexStorage -> IO ()
  reshape_ r t = Dynamic.reshape_ (asDynamic r) (asDynamic t)
  cat_ :: Tensor d -> Tensor d -> Tensor d -> DimVal -> IO ()
  cat_ r a b = Dynamic.cat_ (asDynamic r) (asDynamic a) (asDynamic b)
  catArray_ :: Tensor d -> [Dynamic] -> Int -> DimVal -> IO ()
  catArray_ r = Dynamic.catArray_ (asDynamic r)
  nonzero_ :: IndexTensor d -> Tensor d -> IO ()
  nonzero_ r t = Dynamic.nonzero_ (longAsDynamic r) (asDynamic t)
  tril_ :: Tensor d -> Tensor d -> Integer -> IO ()
  tril_ r t = Dynamic.tril_ (asDynamic r) (asDynamic t)
  triu_ :: Tensor d -> Tensor d -> Integer -> IO ()
  triu_ r t = Dynamic.triu_ (asDynamic r) (asDynamic t)
  diag_ :: Tensor d -> Tensor d -> Int -> IO ()
  diag_ r t = Dynamic.diag_ (asDynamic r) (asDynamic t)
  eye_ :: Tensor d -> Integer -> Integer -> IO ()
  eye_ r = Dynamic.eye_ (asDynamic r)
  trace :: Tensor d -> IO HsAccReal
  trace r = Dynamic.trace (asDynamic r)
  arange_ :: Tensor d -> HsAccReal -> HsAccReal -> HsAccReal -> IO ()
  arange_ r = Dynamic.arange_ (asDynamic r)
  range_ :: Tensor d -> HsAccReal -> HsAccReal -> HsAccReal -> IO ()
  range_ r = Dynamic.range_ (asDynamic r)


