module Torch.Indef.Static.Tensor.Math.CompareT where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.CompareT.Static as Class
import qualified Torch.Class.Tensor.Math.CompareT as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.CompareT ()

instance Class.TensorMathCompareT Tensor where
  ltTensor_ :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  ltTensor_ m t t2 = Dynamic.ltTensor_ (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  leTensor_ :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  leTensor_ m t t2 = Dynamic.leTensor_ (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  gtTensor_ :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  gtTensor_ m t t2 = Dynamic.gtTensor_ (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  geTensor_ :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  geTensor_ m t t2 = Dynamic.geTensor_ (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  neTensor_ :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  neTensor_ m t t2 = Dynamic.neTensor_ (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  eqTensor_ :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  eqTensor_ m t t2 = Dynamic.eqTensor_ (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  ltTensorT_ :: Tensor d -> Tensor d -> Tensor d -> IO ()
  ltTensorT_ r t t2 = Dynamic.ltTensorT_ (asDynamic r) (asDynamic t) (asDynamic t2)

  leTensorT_ :: Tensor d -> Tensor d -> Tensor d -> IO ()
  leTensorT_ r t t2 = Dynamic.leTensorT_ (asDynamic r) (asDynamic t) (asDynamic t2)

  gtTensorT_ :: Tensor d -> Tensor d -> Tensor d -> IO ()
  gtTensorT_ r t t2 = Dynamic.gtTensorT_ (asDynamic r) (asDynamic t) (asDynamic t2)

  geTensorT_ :: Tensor d -> Tensor d -> Tensor d -> IO ()
  geTensorT_ r t t2 = Dynamic.geTensorT_ (asDynamic r) (asDynamic t) (asDynamic t2)

  neTensorT_ :: Tensor d -> Tensor d -> Tensor d -> IO ()
  neTensorT_ r t t2 = Dynamic.neTensorT_ (asDynamic r) (asDynamic t) (asDynamic t2)

  eqTensorT_ :: Tensor d -> Tensor d -> Tensor d -> IO ()
  eqTensorT_ r t t2 = Dynamic.eqTensorT_ (asDynamic r) (asDynamic t) (asDynamic t2)

