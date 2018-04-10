module Torch.Class.Tensor.Math.CompareT where

import Torch.Class.Types

class TensorMathCompareT t where
  _ltTensor    :: MaskDynamic t -> t -> t -> IO ()
  _leTensor    :: MaskDynamic t -> t -> t -> IO ()
  _gtTensor    :: MaskDynamic t -> t -> t -> IO ()
  _geTensor    :: MaskDynamic t -> t -> t -> IO ()
  _neTensor    :: MaskDynamic t -> t -> t -> IO ()
  _eqTensor    :: MaskDynamic t -> t -> t -> IO ()

  _ltTensorT   :: t -> t -> t -> IO ()
  _leTensorT   :: t -> t -> t -> IO ()
  _gtTensorT   :: t -> t -> t -> IO ()
  _geTensorT   :: t -> t -> t -> IO ()
  _neTensorT   :: t -> t -> t -> IO ()
  _eqTensorT   :: t -> t -> t -> IO ()


