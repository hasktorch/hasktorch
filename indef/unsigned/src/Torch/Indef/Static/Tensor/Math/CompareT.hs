module Torch.Indef.Static.Tensor.Math.CompareT where

import Torch.Dimensions
import qualified Torch.Class.Tensor.Math.CompareT.Static as Class
import qualified Torch.Class.Tensor.Math.CompareT as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.CompareT ()
import Torch.Indef.Static.Tensor ()

instance Class.TensorMathCompareT Tensor where
  _ltTensor :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  _ltTensor m t t2 = Dynamic._ltTensor (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  _leTensor :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  _leTensor m t t2 = Dynamic._leTensor (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  _gtTensor :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  _gtTensor m t t2 = Dynamic._gtTensor (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  _geTensor :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  _geTensor m t t2 = Dynamic._geTensor (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  _neTensor :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  _neTensor m t t2 = Dynamic._neTensor (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  _eqTensor :: ByteTensor n -> Tensor d -> Tensor d -> IO ()
  _eqTensor m t t2 = Dynamic._eqTensor (byteAsDynamic m) (asDynamic t) (asDynamic t2)

  _ltTensorT :: Tensor d -> Tensor d -> Tensor d -> IO ()
  _ltTensorT r t t2 = Dynamic._ltTensorT (asDynamic r) (asDynamic t) (asDynamic t2)

  _leTensorT :: Tensor d -> Tensor d -> Tensor d -> IO ()
  _leTensorT r t t2 = Dynamic._leTensorT (asDynamic r) (asDynamic t) (asDynamic t2)

  _gtTensorT :: Tensor d -> Tensor d -> Tensor d -> IO ()
  _gtTensorT r t t2 = Dynamic._gtTensorT (asDynamic r) (asDynamic t) (asDynamic t2)

  _geTensorT :: Tensor d -> Tensor d -> Tensor d -> IO ()
  _geTensorT r t t2 = Dynamic._geTensorT (asDynamic r) (asDynamic t) (asDynamic t2)

  _neTensorT :: Tensor d -> Tensor d -> Tensor d -> IO ()
  _neTensorT r t t2 = Dynamic._neTensorT (asDynamic r) (asDynamic t) (asDynamic t2)

  _eqTensorT :: Tensor d -> Tensor d -> Tensor d -> IO ()
  _eqTensorT r t t2 = Dynamic._eqTensorT (asDynamic r) (asDynamic t) (asDynamic t2)

