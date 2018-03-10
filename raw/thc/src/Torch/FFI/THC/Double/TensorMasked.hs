{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMasked
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
foreign import ccall "THCTensorMasked.h THCDoubleTensor_maskedFill"
  c_maskedFill :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> CDouble -> IO ()

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCDoubleTensor_maskedFillByte"
  c_maskedFillByte :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> CDouble -> IO ()

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCDoubleTensor_maskedCopy"
  c_maskedCopy :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCDoubleTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCDoubleTensor_maskedSelect"
  c_maskedSelect :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCDoubleTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCDoubleTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> CDouble -> IO ())

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCDoubleTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> CDouble -> IO ())

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCDoubleTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCDoubleTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCDoubleTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> IO ())

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCDoubleTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> IO ())