module Torch.Class.Tensor.Math.CompareT where

import Torch.Class.Types

class TensorMathCompareT t where
  ltTensor_    :: MaskTensor t -> t -> t -> io ()
  leTensor_    :: MaskTensor t -> t -> t -> io ()
  gtTensor_    :: MaskTensor t -> t -> t -> io ()
  geTensor_    :: MaskTensor t -> t -> t -> io ()
  neTensor_    :: MaskTensor t -> t -> t -> io ()
  eqTensor_    :: MaskTensor t -> t -> t -> io ()


  ltTensorT_   :: t -> t -> t -> io ()
  leTensorT_   :: t -> t -> t -> io ()
  gtTensorT_   :: t -> t -> t -> io ()
  geTensorT_   :: t -> t -> t -> io ()
  neTensorT_   :: t -> t -> t -> io ()
  eqTensorT_   :: t -> t -> t -> io ()


