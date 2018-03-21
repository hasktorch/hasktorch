{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.TensorMasked where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_maskedFill :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCudaShortTensor_maskedFill"
  c_maskedFill :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaByteTensor -> CShort -> IO ()

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCudaShortTensor_maskedFillByte"
  c_maskedFillByte :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THByteTensor -> CShort -> IO ()

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCudaShortTensor_maskedCopy"
  c_maskedCopy :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCudaShortTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THByteTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCudaShortTensor_maskedSelect"
  c_maskedSelect :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCudaShortTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> Ptr C'THByteTensor -> IO ()

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCudaShortTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaByteTensor -> CShort -> IO ())

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCudaShortTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THByteTensor -> CShort -> IO ())

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCudaShortTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaShortTensor -> IO ())

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCudaShortTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THByteTensor -> Ptr C'THCudaShortTensor -> IO ())

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCudaShortTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCudaShortTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> Ptr C'THByteTensor -> IO ())