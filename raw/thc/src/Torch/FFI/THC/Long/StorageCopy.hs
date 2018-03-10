{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Long.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_rawCopy"
  c_rawCopy :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CLong -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copy"
  c_copy :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyByte"
  c_copyByte :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyChar"
  c_copyChar :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyShort"
  c_copyShort :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyInt"
  c_copyInt :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyLong"
  c_copyLong :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyFloat"
  c_copyFloat :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyDouble"
  c_copyDouble :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyHalf"
  c_copyHalf :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyCuda"
  c_copyCuda :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCLongStorage_copyCPU"
  c_copyCPU :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CLong -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copy"
  p_copy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCLongStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> Ptr CTHCudaLongStorage -> IO ())