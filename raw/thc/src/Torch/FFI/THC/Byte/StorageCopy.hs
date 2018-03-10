{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Byte.StorageCopy
  ( c_rawCopy
  , c_copy
  , c_copyByte
  , c_copyChar
  , c_copyShort
  , c_copyInt
  , c_copyLong
  , c_copyFloat
  , c_copyDouble
  , c_copyHalf
  , c_copyCuda
  , c_copyCPU
  , p_rawCopy
  , p_copy
  , p_copyByte
  , p_copyChar
  , p_copyShort
  , p_copyInt
  , p_copyLong
  , p_copyFloat
  , p_copyDouble
  , p_copyHalf
  , p_copyCuda
  , p_copyCPU
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_rawCopy"
  c_rawCopy :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CUChar -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copy"
  c_copy :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyByte"
  c_copyByte :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyChar"
  c_copyChar :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyShort"
  c_copyShort :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyInt"
  c_copyInt :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyLong"
  c_copyLong :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyFloat"
  c_copyFloat :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyDouble"
  c_copyDouble :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyHalf"
  c_copyHalf :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyCuda"
  c_copyCuda :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCByteStorage_copyCPU"
  c_copyCPU :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CUChar -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copy"
  p_copy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaByteStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaByteStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCByteStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> Ptr CTHCudaByteStorage -> IO ())