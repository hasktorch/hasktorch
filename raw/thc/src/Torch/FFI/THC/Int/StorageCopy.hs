{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_rawCopy"
  c_rawCopy :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr CInt -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copy"
  c_copy :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCIntStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyByte"
  c_copyByte :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyChar"
  c_copyChar :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyShort"
  c_copyShort :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyInt"
  c_copyInt :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyLong"
  c_copyLong :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyFloat"
  c_copyFloat :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyDouble"
  c_copyDouble :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyHalf"
  c_copyHalf :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyCuda"
  c_copyCuda :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCIntStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCIntStorage_copyCPU"
  c_copyCPU :: Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCIntStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr CInt -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copy"
  p_copy :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCIntStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyByte"
  p_copyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyChar"
  p_copyChar :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyShort"
  p_copyShort :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyInt"
  p_copyInt :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyLong"
  p_copyLong :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCIntStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCIntStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr C'THCState -> Ptr C'THCIntStorage -> Ptr C'THCIntStorage -> IO ())