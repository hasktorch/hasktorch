{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.TensorMasked
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
foreign import ccall "THCTensorMasked.h THHalfTensor_maskedFill"
  c_maskedFill :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> CTHHalf -> IO (())

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THHalfTensor_maskedFillByte"
  c_maskedFillByte :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> CTHHalf -> IO (())

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THHalfTensor_maskedCopy"
  c_maskedCopy :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHHalfTensor) -> IO (())

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THHalfTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHHalfTensor) -> IO (())

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THHalfTensor_maskedSelect"
  c_maskedSelect :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THHalfTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THHalfTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> CTHHalf -> IO (()))

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THHalfTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> CTHHalf -> IO (()))

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THHalfTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHHalfTensor) -> IO (()))

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THHalfTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHHalfTensor) -> IO (()))

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THHalfTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THHalfTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHHalfTensor) -> Ptr (CTHByteTensor) -> IO (()))