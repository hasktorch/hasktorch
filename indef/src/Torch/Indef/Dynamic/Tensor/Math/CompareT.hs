module Torch.Indef.Dynamic.Tensor.Math.CompareT
  ( ltTensor, ltTensorT, ltTensorT_
  , leTensor, leTensorT, leTensorT_
  , gtTensor, gtTensorT, gtTensorT_
  , geTensor, geTensorT, geTensorT_
  , neTensor, neTensorT, neTensorT_
  , eqTensor, eqTensorT, eqTensorT_
  ) where

import System.IO.Unsafe
import qualified Torch.Sig.Tensor.Math.CompareT as Sig

import Torch.Indef.Mask
import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor

_ltTensorT :: Dynamic -> Dynamic -> Dynamic -> IO ()
_ltTensorT = shuffle3 with3DynamicState Sig.c_ltTensorT

_leTensorT :: Dynamic -> Dynamic -> Dynamic -> IO ()
_leTensorT = shuffle3 with3DynamicState Sig.c_leTensorT

_gtTensorT :: Dynamic -> Dynamic -> Dynamic -> IO ()
_gtTensorT = shuffle3 with3DynamicState Sig.c_gtTensorT

_geTensorT :: Dynamic -> Dynamic -> Dynamic -> IO ()
_geTensorT = shuffle3 with3DynamicState Sig.c_geTensorT

_neTensorT :: Dynamic -> Dynamic -> Dynamic -> IO ()
_neTensorT = shuffle3 with3DynamicState Sig.c_neTensorT

_eqTensorT :: Dynamic -> Dynamic -> Dynamic -> IO ()
_eqTensorT = shuffle3 with3DynamicState Sig.c_eqTensorT

compareTensorOp
  :: (Ptr CState -> Ptr CByteTensor -> Ptr CTensor -> Ptr CTensor -> IO ())
  -> Dynamic -> Dynamic -> MaskDynamic
compareTensorOp op t0 t1 = unsafeDupablePerformIO $ do
  SomeDims d <- getDims t0
  let bt = newMaskDyn d
  with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> op s' bt' t0' t1'
  pure bt

ltTensor, leTensor, gtTensor, geTensor, neTensor, eqTensor
  :: Dynamic -> Dynamic -> MaskDynamic
ltTensor = compareTensorOp Sig.c_ltTensor
leTensor = compareTensorOp Sig.c_leTensor
gtTensor = compareTensorOp Sig.c_gtTensor
geTensor = compareTensorOp Sig.c_geTensor
neTensor = compareTensorOp Sig.c_neTensor
eqTensor = compareTensorOp Sig.c_eqTensor

ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: Dynamic -> Dynamic -> Dynamic
ltTensorT  a b = unsafeDupablePerformIO $ withEmpty a $ \r -> _ltTensorT r a b
leTensorT  a b = unsafeDupablePerformIO $ withEmpty a $ \r -> _leTensorT r a b
gtTensorT  a b = unsafeDupablePerformIO $ withEmpty a $ \r -> _gtTensorT r a b
geTensorT  a b = unsafeDupablePerformIO $ withEmpty a $ \r -> _geTensorT r a b
neTensorT  a b = unsafeDupablePerformIO $ withEmpty a $ \r -> _neTensorT r a b
eqTensorT  a b = unsafeDupablePerformIO $ withEmpty a $ \r -> _eqTensorT r a b

ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: Dynamic -> Dynamic -> IO Dynamic
ltTensorT_ a b = _ltTensorT a a b >> pure a
leTensorT_ a b = _leTensorT a a b >> pure a
gtTensorT_ a b = _gtTensorT a a b >> pure a
geTensorT_ a b = _geTensorT a a b >> pure a
neTensorT_ a b = _neTensorT a a b >> pure a
eqTensorT_ a b = _eqTensorT a a b >> pure a

