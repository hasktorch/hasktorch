module Torch.Indef.Static.Tensor.Math.CompareT where

import Torch.Dimensions

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.CompareT as Dynamic

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

-- compareTensors
--   :: (Dimensions d)
--   => (MaskTensor t d -> t d -> t d -> IO ())
--   -> t d -> t d -> IO (MaskTensor t d)
-- compareTensors op a b = do
--   r <- Dynamic.empty
--   op (asStatic r) a b
--   pure (asStatic r)
-- 
-- ltTensor, leTensor, gtTensor, geTensor, neTensor, eqTensor
--   :: (Dimensions d)
--   => Tensor d -> Tensor d -> IO (MaskTensor d)
-- ltTensor = compareTensors _ltTensor
-- leTensor = compareTensors _leTensor
-- gtTensor = compareTensors _gtTensor
-- geTensor = compareTensors _geTensor
-- neTensor = compareTensors _neTensor
-- eqTensor = compareTensors _eqTensor

ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: Dimensions d => Tensor d -> Tensor d -> IO (Tensor d)
ltTensorT a b = withEmpty $ \r -> _ltTensorT r a b
leTensorT a b = withEmpty $ \r -> _leTensorT r a b
gtTensorT a b = withEmpty $ \r -> _gtTensorT r a b
geTensorT a b = withEmpty $ \r -> _geTensorT r a b
neTensorT a b = withEmpty $ \r -> _neTensorT r a b
eqTensorT a b = withEmpty $ \r -> _eqTensorT r a b

ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
ltTensorT_ a b = _ltTensorT a a b >> pure a
leTensorT_ a b = _leTensorT a a b >> pure a
gtTensorT_ a b = _gtTensorT a a b >> pure a
geTensorT_ a b = _geTensorT a a b >> pure a
neTensorT_ a b = _neTensorT a a b >> pure a
eqTensorT_ a b = _eqTensorT a a b >> pure a

