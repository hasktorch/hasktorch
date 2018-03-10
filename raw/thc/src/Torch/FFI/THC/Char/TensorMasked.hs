{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.TensorMasked where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_maskedFill :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCCharTensor_maskedFill"
  c_maskedFill :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> CChar -> IO ()

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCCharTensor_maskedFillByte"
  c_maskedFillByte :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> CChar -> IO ()

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCCharTensor_maskedCopy"
  c_maskedCopy :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCCharTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCCharTensor_maskedSelect"
  c_maskedSelect :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCCharTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCCharTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> CChar -> IO ())

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCCharTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> CChar -> IO ())

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCCharTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaCharTensor -> IO ())

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCCharTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaCharTensor -> IO ())

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCCharTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> IO ())

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCCharTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> Ptr CTHCudaByteTensor -> IO ())