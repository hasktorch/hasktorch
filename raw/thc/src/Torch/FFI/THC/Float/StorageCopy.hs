{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_rawCopy"
  c_rawCopy :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CFloat -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copy"
  c_copy :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyByte"
  c_copyByte :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyChar"
  c_copyChar :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyShort"
  c_copyShort :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyInt"
  c_copyInt :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyLong"
  c_copyLong :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyFloat"
  c_copyFloat :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyDouble"
  c_copyDouble :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyHalf"
  c_copyHalf :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyCuda"
  c_copyCuda :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCFloatStorage_copyCPU"
  c_copyCPU :: Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CFloat -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copy"
  p_copy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCFloatStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatStorage -> Ptr CTHCudaFloatStorage -> IO ())