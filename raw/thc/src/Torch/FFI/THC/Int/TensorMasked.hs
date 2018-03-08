{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorMasked
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
foreign import ccall "THCTensorMasked.h THIntTensor_maskedFill"
  c_maskedFill :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THIntTensor_maskedFillByte"
  c_maskedFillByte :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (())

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THIntTensor_maskedCopy"
  c_maskedCopy :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THIntTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THIntTensor_maskedSelect"
  c_maskedSelect :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THIntTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THIntTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THIntTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> CInt -> IO (()))

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THIntTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THIntTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THIntTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THIntTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHByteTensor) -> IO (()))