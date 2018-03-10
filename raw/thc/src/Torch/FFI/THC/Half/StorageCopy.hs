{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_rawCopy"
  c_rawCopy :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHHalf -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copy"
  c_copy :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyByte"
  c_copyByte :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyChar"
  c_copyChar :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyShort"
  c_copyShort :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyInt"
  c_copyInt :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyLong"
  c_copyLong :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyFloat"
  c_copyFloat :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyDouble"
  c_copyDouble :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyHalf"
  c_copyHalf :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyCuda"
  c_copyCuda :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCHalfStorage_copyCPU"
  c_copyCPU :: Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHHalf -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copy"
  p_copy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCHalfStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfStorage -> Ptr CTHCudaHalfStorage -> IO ())