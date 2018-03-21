{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Long.TensorMasked where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_maskedFill :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCudaLongTensor_maskedFill"
  c_maskedFill :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaByteTensor -> CLong -> IO ()

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCudaLongTensor_maskedFillByte"
  c_maskedFillByte :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THByteTensor -> CLong -> IO ()

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCudaLongTensor_maskedCopy"
  c_maskedCopy :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaLongTensor -> IO ()

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCudaLongTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THByteTensor -> Ptr C'THCudaLongTensor -> IO ()

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCudaLongTensor_maskedSelect"
  c_maskedSelect :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCudaLongTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THByteTensor -> IO ()

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCudaLongTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaByteTensor -> CLong -> IO ())

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCudaLongTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THByteTensor -> CLong -> IO ())

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCudaLongTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaLongTensor -> IO ())

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCudaLongTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THByteTensor -> Ptr C'THCudaLongTensor -> IO ())

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCudaLongTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCudaLongTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THByteTensor -> IO ())