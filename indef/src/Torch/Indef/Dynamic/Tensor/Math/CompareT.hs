module Torch.Indef.Dynamic.Tensor.Math.CompareT where

import qualified Torch.Sig.Tensor.Math.CompareT as Sig

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor

_ltTensor :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
_ltTensor bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_ltTensor s' bt' t0' t1'

_leTensor :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
_leTensor bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_leTensor s' bt' t0' t1'

_gtTensor :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
_gtTensor bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_gtTensor s' bt' t0' t1'

_geTensor :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
_geTensor bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_geTensor s' bt' t0' t1'

_neTensor :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
_neTensor bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_neTensor s' bt' t0' t1'

_eqTensor :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
_eqTensor bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_eqTensor s' bt' t0' t1'

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

-- ltTensor, leTensor, gtTensor, geTensor, neTensor, eqTensor
--   :: Dynamic -> Dynamic -> IO MaskDynamic
-- ltTensor  a b = getDims a >>= new' >>= \r -> _ltTensor r a b >> pure r
-- leTensor  a b = getDims a >>= new' >>= \r -> _leTensor r a b >> pure r
-- gtTensor  a b = getDims a >>= new' >>= \r -> _gtTensor r a b >> pure r
-- geTensor  a b = getDims a >>= new' >>= \r -> _geTensor r a b >> pure r
-- neTensor  a b = getDims a >>= new' >>= \r -> _neTensor r a b >> pure r
-- eqTensor  a b = getDims a >>= new' >>= \r -> _eqTensor r a b >> pure r

ltTensorT, leTensorT, gtTensorT, geTensorT, neTensorT, eqTensorT
  :: Dynamic -> Dynamic -> IO Dynamic
ltTensorT  a b = withEmpty a $ \r -> _ltTensorT r a b
leTensorT  a b = withEmpty a $ \r -> _leTensorT r a b
gtTensorT  a b = withEmpty a $ \r -> _gtTensorT r a b
geTensorT  a b = withEmpty a $ \r -> _geTensorT r a b
neTensorT  a b = withEmpty a $ \r -> _neTensorT r a b
eqTensorT  a b = withEmpty a $ \r -> _eqTensorT r a b

ltTensorT_, leTensorT_, gtTensorT_, geTensorT_, neTensorT_, eqTensorT_
  :: Dynamic -> Dynamic -> IO Dynamic
ltTensorT_ a b = _ltTensorT a a b >> pure a
leTensorT_ a b = _leTensorT a a b >> pure a
gtTensorT_ a b = _gtTensorT a a b >> pure a
geTensorT_ a b = _geTensorT a a b >> pure a
neTensorT_ a b = _neTensorT a a b >> pure a
eqTensorT_ a b = _eqTensorT a a b >> pure a

