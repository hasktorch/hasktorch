{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Class.Tensor.Math.CompareT.Static where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import GHC.TypeLits
import Torch.Dimensions
import qualified Torch.Class.Tensor as Dynamic

class IsTensor t => TensorMathCompareT t where
  _ltTensor :: Dimensions d => MaskTensor t d -> t d -> t d -> IO ()
  _leTensor :: Dimensions d => MaskTensor t d -> t d -> t d -> IO ()
  _gtTensor :: Dimensions d => MaskTensor t d -> t d -> t d -> IO ()
  _geTensor :: Dimensions d => MaskTensor t d -> t d -> t d -> IO ()
  _neTensor :: Dimensions d => MaskTensor t d -> t d -> t d -> IO ()
  _eqTensor :: Dimensions d => MaskTensor t d -> t d -> t d -> IO ()

  _ltTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _leTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _gtTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _geTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _neTensorT :: Dimensions d => t d -> t d -> t d -> IO ()
  _eqTensorT :: Dimensions d => t d -> t d -> t d -> IO ()


compareTensors
  :: (Dimensions d, Dynamic.IsTensor (MaskTensor t d), TensorMathCompareT t)
  => (Dynamic.IsTensor (AsDynamic (MaskTensor t)))
  => (IsStatic (MaskTensor t))
  => (MaskTensor t d -> t d -> t d -> IO ())
  -> t d -> t d -> IO (MaskTensor t d)
compareTensors op a b = do
  r <- Dynamic.empty
  op (asStatic r) a b
  pure (asStatic r)

ltTensor, leTensor, gtTensor, geTensor, neTensor, eqTensor
  :: (Dimensions d, Dynamic.IsTensor (MaskTensor t d), TensorMathCompareT t)
  => (Dynamic.IsTensor (AsDynamic (MaskTensor t)))
  => (IsStatic (MaskTensor t))
  => t d -> t d -> IO (MaskTensor t d)
ltTensor = compareTensors _ltTensor
leTensor = compareTensors _leTensor
gtTensor = compareTensors _gtTensor
geTensor = compareTensors _geTensor
neTensor = compareTensors _neTensor
eqTensor = compareTensors _eqTensor

ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: (IsTensor t, Dimensions d, TensorMathCompareT t)
  => t d -> t d -> IO (t d)
ltTensorT a b = withEmpty $ \r -> _ltTensorT r a b
leTensorT a b = withEmpty $ \r -> _leTensorT r a b
gtTensorT a b = withEmpty $ \r -> _gtTensorT r a b
geTensorT a b = withEmpty $ \r -> _geTensorT r a b
neTensorT a b = withEmpty $ \r -> _neTensorT r a b
eqTensorT a b = withEmpty $ \r -> _eqTensorT r a b

ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: (Dimensions d, TensorMathCompareT t)
  => t d -> t d -> IO (t d)
ltTensorT_ a b = _ltTensorT a a b >> pure a
leTensorT_ a b = _leTensorT a a b >> pure a
gtTensorT_ a b = _gtTensorT a a b >> pure a
geTensorT_ a b = _geTensorT a a b >> pure a
neTensorT_ a b = _neTensorT a a b >> pure a
eqTensorT_ a b = _eqTensorT a a b >> pure a

