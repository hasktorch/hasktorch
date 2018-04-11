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
ltTensor  a b = withEmpty $ \r -> _ltTensor r a b
leTensor  a b = withEmpty $ \r -> _leTensor r a b
gtTensor  a b = withEmpty $ \r -> _gtTensor r a b
geTensor  a b = withEmpty $ \r -> _geTensor r a b
neTensor  a b = withEmpty $ \r -> _neTensor r a b
eqTensor  a b = withEmpty $ \r -> _eqTensor r a b

ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: TensorMathCompareT t => t -> t -> IO t
ltTensorT  a b = withEmpty $ \r -> _ltTensorT r a b
leTensorT  a b = withEmpty $ \r -> _leTensorT r a b
gtTensorT  a b = withEmpty $ \r -> _gtTensorT r a b
geTensorT  a b = withEmpty $ \r -> _geTensorT r a b
neTensorT  a b = withEmpty $ \r -> _neTensorT r a b
eqTensorT  a b = withEmpty $ \r -> _eqTensorT r a b

ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: TensorMathCompareT t => t -> t -> IO t
ltTensorT_ a b = _ltTensorT a a b >> pure a
leTensorT_ a b = _leTensorT a a b >> pure a
gtTensorT_ a b = _gtTensorT a a b >> pure a
geTensorT_ a b = _geTensorT a a b >> pure a
neTensorT_ a b = _neTensorT a a b >> pure a
eqTensorT_ a b = _eqTensorT a a b >> pure a

