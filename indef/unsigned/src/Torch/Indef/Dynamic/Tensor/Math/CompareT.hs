module Torch.Indef.Dynamic.Tensor.Math.CompareT where

import Torch.Class.Tensor.Math.CompareT
import qualified Torch.Sig.Tensor.Math.CompareT as Sig

import Torch.Indef.Types

instance TensorMathCompareT Dynamic where
  ltTensor_ :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
  ltTensor_ bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_ltTensor s' bt' t0' t1'

  leTensor_ :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
  leTensor_ bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_leTensor s' bt' t0' t1'

  gtTensor_ :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
  gtTensor_ bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_gtTensor s' bt' t0' t1'

  geTensor_ :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
  geTensor_ bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_geTensor s' bt' t0' t1'

  neTensor_ :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
  neTensor_ bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_neTensor s' bt' t0' t1'

  eqTensor_ :: MaskDynamic -> Dynamic -> Dynamic -> IO ()
  eqTensor_ bt t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withMask bt $ \bt' -> Sig.c_eqTensor s' bt' t0' t1'

  ltTensorT_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  ltTensorT_ = shuffle3 with3DynamicState Sig.c_ltTensorT

  leTensorT_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  leTensorT_ = shuffle3 with3DynamicState Sig.c_leTensorT

  gtTensorT_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  gtTensorT_ = shuffle3 with3DynamicState Sig.c_gtTensorT

  geTensorT_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  geTensorT_ = shuffle3 with3DynamicState Sig.c_geTensorT

  neTensorT_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  neTensorT_ = shuffle3 with3DynamicState Sig.c_neTensorT

  eqTensorT_ :: Dynamic -> Dynamic -> Dynamic -> IO ()
  eqTensorT_ = shuffle3 with3DynamicState Sig.c_eqTensorT
