{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorMathMagma
  ( c_gesv
  , c_gels
  , c_syev
  , c_geev
  , c_gesvd
  , c_gesvd2
  , c_getri
  , c_potri
  , c_potrf
  , c_potrs
  , c_geqrf
  , c_qr
  , p_gesv
  , p_gels
  , p_syev
  , p_geev
  , p_gesvd
  , p_gesvd2
  , p_getri
  , p_potri
  , p_potrf
  , p_potrs
  , p_geqrf
  , p_qr
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gesv :  state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_gesv"
  c_gesv :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_gels :  state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_gels"
  c_gels :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_syev :  state re_ rv_ a_ jobz uplo -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_syev"
  c_syev :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> Ptr (CChar) -> IO (())

-- | c_geev :  state re_ rv_ a_ jobvr -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_geev"
  c_geev :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_gesvd :  state ru_ rs_ rv_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_gesvd"
  c_gesvd :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_gesvd2 :  state ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_gesvd2"
  c_gesvd2 :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_getri :  state ra_ a -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_getri"
  c_getri :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_potri :  state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_potri"
  c_potri :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_potrf :  state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_potrf"
  c_potrf :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_potrs :  state rb_ a b uplo -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_potrs"
  c_potrs :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_geqrf :  state ra_ rtau_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_geqrf"
  c_geqrf :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_qr :  state rq_ rr_ a -> void
foreign import ccall "THCTensorMathMagma.h THDoubleTensor_qr"
  c_qr :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | p_gesv : Pointer to function : state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_gesv"
  p_gesv :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_gels : Pointer to function : state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_gels"
  p_gels :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_syev : Pointer to function : state re_ rv_ a_ jobz uplo -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_syev"
  p_syev :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> Ptr (CChar) -> IO (()))

-- | p_geev : Pointer to function : state re_ rv_ a_ jobvr -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_geev"
  p_geev :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_gesvd : Pointer to function : state ru_ rs_ rv_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_gesvd"
  p_gesvd :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_gesvd2 : Pointer to function : state ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_gesvd2"
  p_gesvd2 :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_getri : Pointer to function : state ra_ a -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_getri"
  p_getri :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_potri : Pointer to function : state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_potri"
  p_potri :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_potrf : Pointer to function : state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_potrf"
  p_potrf :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_potrs : Pointer to function : state rb_ a b uplo -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_potrs"
  p_potrs :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_geqrf : Pointer to function : state ra_ rtau_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_geqrf"
  p_geqrf :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_qr : Pointer to function : state rq_ rr_ a -> void
foreign import ccall "THCTensorMathMagma.h &THDoubleTensor_qr"
  p_qr :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))