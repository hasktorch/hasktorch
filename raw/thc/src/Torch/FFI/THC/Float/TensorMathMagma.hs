{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMathMagma where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_gesv :  state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_gesv"
  c_gesv :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_gels :  state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_gels"
  c_gels :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_syev :  state re_ rv_ a_ jobz uplo -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_syev"
  c_syev :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_geev :  state re_ rv_ a_ jobvr -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_geev"
  c_geev :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_gesvd :  state ru_ rs_ rv_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_gesvd"
  c_gesvd :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_gesvd2 :  state ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_gesvd2"
  c_gesvd2 :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_getri :  state ra_ a -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_getri"
  c_getri :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_potri :  state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_potri"
  c_potri :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_potrf :  state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_potrf"
  c_potrf :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_potrs :  state rb_ a b uplo -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_potrs"
  c_potrs :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ()

-- | c_geqrf :  state ra_ rtau_ a_ -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_geqrf"
  c_geqrf :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_qr :  state rq_ rr_ a -> void
foreign import ccall "THCTensorMathMagma.h THCudaFloatTensor_qr"
  c_qr :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | p_gesv : Pointer to function : state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_gesv"
  p_gesv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_gels : Pointer to function : state rb_ ra_ b_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_gels"
  p_gels :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_syev : Pointer to function : state re_ rv_ a_ jobz uplo -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_syev"
  p_syev :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> Ptr CChar -> IO ())

-- | p_geev : Pointer to function : state re_ rv_ a_ jobvr -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_geev"
  p_geev :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_gesvd : Pointer to function : state ru_ rs_ rv_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_gesvd"
  p_gesvd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_gesvd2 : Pointer to function : state ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_gesvd2"
  p_gesvd2 :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_getri : Pointer to function : state ra_ a -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_getri"
  p_getri :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_potri : Pointer to function : state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_potri"
  p_potri :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_potrf : Pointer to function : state ra_ a uplo -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_potrf"
  p_potrf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_potrs : Pointer to function : state rb_ a b uplo -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_potrs"
  p_potrs :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr CChar -> IO ())

-- | p_geqrf : Pointer to function : state ra_ rtau_ a_ -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_geqrf"
  p_geqrf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_qr : Pointer to function : state rq_ rr_ a -> void
foreign import ccall "THCTensorMathMagma.h &THCudaFloatTensor_qr"
  p_qr :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())