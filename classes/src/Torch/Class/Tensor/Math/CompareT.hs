module Torch.Class.Tensor.Math.CompareT where

import Torch.Class.Types

class TensorMathCompareT t where
  ltTensor_    :: MaskDynamic t -> t -> t -> IO ()
  leTensor_    :: MaskDynamic t -> t -> t -> IO ()
  gtTensor_    :: MaskDynamic t -> t -> t -> IO ()
  geTensor_    :: MaskDynamic t -> t -> t -> IO ()
  neTensor_    :: MaskDynamic t -> t -> t -> IO ()
  eqTensor_    :: MaskDynamic t -> t -> t -> IO ()

  ltTensorT_   :: t -> t -> t -> IO ()
  leTensorT_   :: t -> t -> t -> IO ()
  gtTensorT_   :: t -> t -> t -> IO ()
  geTensorT_   :: t -> t -> t -> IO ()
  neTensorT_   :: t -> t -> t -> IO ()
  eqTensorT_   :: t -> t -> t -> IO ()


