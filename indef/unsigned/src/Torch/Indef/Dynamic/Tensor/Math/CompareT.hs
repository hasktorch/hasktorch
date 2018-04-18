module Torch.Indef.Dynamic.Tensor.Math.CompareT where

import Torch.Class.Tensor.Math.CompareT
import qualified Torch.Sig.Tensor.Math.CompareT as Sig
import Torch.Indef.Dynamic.Tensor ()

import Torch.Indef.Types

instance TensorMathCompareT Dynamic where
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
