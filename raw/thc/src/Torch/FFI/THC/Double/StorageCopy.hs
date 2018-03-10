{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.StorageCopy
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
foreign import ccall "THCStorageCopy.h THCDoubleStorage_rawCopy"
  c_rawCopy :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CDouble -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copy"
  c_copy :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyByte"
  c_copyByte :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyChar"
  c_copyChar :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyShort"
  c_copyShort :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyInt"
  c_copyInt :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyLong"
  c_copyLong :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyFloat"
  c_copyFloat :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyDouble"
  c_copyDouble :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyHalf"
  c_copyHalf :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyCuda"
  c_copyCuda :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyCPU"
  c_copyCPU :: Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaDoubleStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CDouble -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copy"
  p_copy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaDoubleStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleStorage -> Ptr CTHCudaDoubleStorage -> IO ())