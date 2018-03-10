{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorMasked where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_maskedFill :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCHalfTensor_maskedFill"
  c_maskedFill :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> CTHHalf -> IO ()

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCHalfTensor_maskedFillByte"
  c_maskedFillByte :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> CTHHalf -> IO ()

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCHalfTensor_maskedCopy"
  c_maskedCopy :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCHalfTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCHalfTensor_maskedSelect"
  c_maskedSelect :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCHalfTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCHalfTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> CTHHalf -> IO ())

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCHalfTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> CTHHalf -> IO ())

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCHalfTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCHalfTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCHalfTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> IO ())

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCHalfTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaByteTensor -> IO ())