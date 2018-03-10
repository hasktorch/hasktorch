{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorMasked where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_maskedFill :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedFill"
  c_maskedFill :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> CInt -> IO ()

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedFillByte"
  c_maskedFillByte :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> CInt -> IO ()

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedCopy"
  c_maskedCopy :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaIntTensor -> IO ()

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaIntTensor -> IO ()

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedSelect"
  c_maskedSelect :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> CInt -> IO ())

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> CInt -> IO ())

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaIntTensor -> IO ())

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaIntTensor -> IO ())

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> IO ())

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> Ptr CTHCudaByteTensor -> IO ())