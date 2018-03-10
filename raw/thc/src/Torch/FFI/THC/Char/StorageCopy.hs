{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_rawCopy"
  c_rawCopy :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CChar -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copy"
  c_copy :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyByte"
  c_copyByte :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyChar"
  c_copyChar :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyShort"
  c_copyShort :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyInt"
  c_copyInt :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyLong"
  c_copyLong :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyFloat"
  c_copyFloat :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyDouble"
  c_copyDouble :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyHalf"
  c_copyHalf :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyCuda"
  c_copyCuda :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCCharStorage_copyCPU"
  c_copyCPU :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CChar -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copy"
  p_copy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaCharStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaCharStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCCharStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> Ptr CTHCudaCharStorage -> IO ())