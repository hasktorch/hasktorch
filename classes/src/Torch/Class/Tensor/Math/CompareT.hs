{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.Tensor.Math.CompareT where

import Torch.Class.Types
import Torch.Class.Tensor

class IsTensor t => TensorMathCompareT t where
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

ltTensor, leTensor, gtTensor, geTensor, neTensor, eqTensor
  :: IsTensor (MaskDynamic t) => TensorMathCompareT t => t -> t -> IO (MaskDynamic t)
ltTensor  a b = getDims a >>= new' >>= \r -> _ltTensor r a b >> pure r
leTensor  a b = getDims a >>= new' >>= \r -> _leTensor r a b >> pure r
gtTensor  a b = getDims a >>= new' >>= \r -> _gtTensor r a b >> pure r
geTensor  a b = getDims a >>= new' >>= \r -> _geTensor r a b >> pure r
neTensor  a b = getDims a >>= new' >>= \r -> _neTensor r a b >> pure r
eqTensor  a b = getDims a >>= new' >>= \r -> _eqTensor r a b >> pure r

ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: TensorMathCompareT t => t -> t -> IO t
ltTensorT  a b = withEmpty a $ \r -> _ltTensorT r a b
leTensorT  a b = withEmpty a $ \r -> _leTensorT r a b
gtTensorT  a b = withEmpty a $ \r -> _gtTensorT r a b
geTensorT  a b = withEmpty a $ \r -> _geTensorT r a b
neTensorT  a b = withEmpty a $ \r -> _neTensorT r a b
eqTensorT  a b = withEmpty a $ \r -> _eqTensorT r a b

ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: TensorMathCompareT t => t -> t -> IO t
ltTensorT_ a b = _ltTensorT a a b >> pure a
leTensorT_ a b = _leTensorT a a b >> pure a
gtTensorT_ a b = _gtTensorT a a b >> pure a
geTensorT_ a b = _geTensorT a a b >> pure a
neTensorT_ a b = _neTensorT a a b >> pure a
eqTensorT_ a b = _eqTensorT a a b >> pure a

