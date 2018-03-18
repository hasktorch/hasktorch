{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_rawCopy"
  c_rawCopy :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr CShort -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copy"
  c_copy :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCShortStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyByte"
  c_copyByte :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyChar"
  c_copyChar :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyShort"
  c_copyShort :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyInt"
  c_copyInt :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyLong"
  c_copyLong :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyFloat"
  c_copyFloat :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyDouble"
  c_copyDouble :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyHalf"
  c_copyHalf :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCHalfStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyCuda"
  c_copyCuda :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCShortStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCShortStorage_copyCPU"
  c_copyCPU :: Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCShortStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr CShort -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copy"
  p_copy :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCShortStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyByte"
  p_copyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyChar"
  p_copyChar :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyShort"
  p_copyShort :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyInt"
  p_copyInt :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyLong"
  p_copyLong :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCHalfStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCShortStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCShortStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr C'THCState -> Ptr C'THCShortStorage -> Ptr C'THCShortStorage -> IO ())