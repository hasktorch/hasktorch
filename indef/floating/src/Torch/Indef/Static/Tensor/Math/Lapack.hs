module Torch.Indef.Static.Tensor.Math.Lapack where

import GHC.Int
import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.Lapack as Dynamic
import qualified Torch.Class.Tensor.Math.Lapack.Static as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Lapack ()

instance Class.TensorMathLapack Tensor where
  getri_ :: Tensor d -> Tensor d' -> IO ()
  getri_ a b = Dynamic.getri_ (asDynamic a) (asDynamic b)

  potri_ :: Tensor d -> Tensor d' -> [Int8] -> IO ()
  potri_ a b = Dynamic.potri_ (asDynamic a) (asDynamic b)

  potrf_ :: Tensor d -> Tensor d' -> [Int8] -> IO ()
  potrf_ a b = Dynamic.potrf_ (asDynamic a) (asDynamic b)

  geqrf_ :: Tensor d -> Tensor d' -> Tensor d'' -> IO ()
  geqrf_ a b c = Dynamic.geqrf_ (asDynamic a) (asDynamic b) (asDynamic c)

  qr_ :: Tensor d -> Tensor d' -> Tensor d'' -> IO ()
  qr_ a b c = Dynamic.qr_ (asDynamic a) (asDynamic b) (asDynamic c)

  geev_ :: Tensor d -> Tensor d' -> Tensor d'' -> [Int8] -> IO ()
  geev_ a b c = Dynamic.geev_ (asDynamic a) (asDynamic b) (asDynamic c)

  potrs_ :: Tensor d -> Tensor d' -> Tensor d'' -> [Int8] -> IO ()
  potrs_ a b c = Dynamic.potrs_ (asDynamic a) (asDynamic b) (asDynamic c)

  syev_ :: Tensor d -> Tensor d' -> Tensor d'' -> [Int8] -> [Int8] -> IO ()
  syev_ a b c = Dynamic.syev_ (asDynamic a) (asDynamic b) (asDynamic c)

  gesv_ :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> IO ()
  gesv_ a b c d = Dynamic.gesv_ (asDynamic a) (asDynamic b) (asDynamic c) (asDynamic d)

  gels_ :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> IO ()
  gels_ a b c d = Dynamic.gels_ (asDynamic a) (asDynamic b) (asDynamic c) (asDynamic d)

  gesvd_ :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> [Int8] -> IO ()
  gesvd_ a b c d = Dynamic.gesvd_ (asDynamic a) (asDynamic b) (asDynamic c) (asDynamic d)

  gesvd2_ :: Tensor d -> Tensor d' -> Tensor d'' -> Tensor d''' -> Tensor d'''' -> [Int8] -> IO ()
  gesvd2_ a b c d e = Dynamic.gesvd2_ (asDynamic a) (asDynamic b) (asDynamic c) (asDynamic d) (asDynamic e)

