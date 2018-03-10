{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMasked
  ( c_maskedFill
  , c_maskedFillByte
  , c_maskedCopy
  , c_maskedCopyByte
  , c_maskedSelect
  , c_maskedSelectByte
  , p_maskedFill
  , p_maskedFillByte
  , p_maskedCopy
  , p_maskedCopyByte
  , p_maskedSelect
  , p_maskedSelectByte
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_maskedFill :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCFloatTensor_maskedFill"
  c_maskedFill :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> CFloat -> IO ()

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCFloatTensor_maskedFillByte"
  c_maskedFillByte :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> CFloat -> IO ()

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCFloatTensor_maskedCopy"
  c_maskedCopy :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCFloatTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCFloatTensor_maskedSelect"
  c_maskedSelect :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCFloatTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCFloatTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> CFloat -> IO ())

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCFloatTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> CFloat -> IO ())

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCFloatTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCFloatTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCFloatTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> IO ())

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCFloatTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaByteTensor -> IO ())