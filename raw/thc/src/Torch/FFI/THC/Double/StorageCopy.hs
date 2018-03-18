{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_rawCopy"
  c_rawCopy :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr CDouble -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copy"
  c_copy :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCDoubleStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyByte"
  c_copyByte :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyChar"
  c_copyChar :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyShort"
  c_copyShort :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyInt"
  c_copyInt :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyLong"
  c_copyLong :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyFloat"
  c_copyFloat :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyDouble"
  c_copyDouble :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyHalf"
  c_copyHalf :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyCuda"
  c_copyCuda :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCDoubleStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCDoubleStorage_copyCPU"
  c_copyCPU :: Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCDoubleStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr CDouble -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copy"
  p_copy :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCDoubleStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyByte"
  p_copyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyChar"
  p_copyChar :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyShort"
  p_copyShort :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyInt"
  p_copyInt :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyLong"
  p_copyLong :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCDoubleStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCDoubleStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr C'THCState -> Ptr C'THCDoubleStorage -> Ptr C'THCDoubleStorage -> IO ())