{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_rawCopy"
  c_rawCopy :: Ptr CTHShortStorage -> Ptr CShort -> IO ()

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copy"
  c_copy :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyByte"
  c_copyByte :: Ptr CTHShortStorage -> Ptr CTHByteStorage -> IO ()

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyChar"
  c_copyChar :: Ptr CTHShortStorage -> Ptr CTHCharStorage -> IO ()

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyShort"
  c_copyShort :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyInt"
  c_copyInt :: Ptr CTHShortStorage -> Ptr CTHIntStorage -> IO ()

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyLong"
  c_copyLong :: Ptr CTHShortStorage -> Ptr CTHLongStorage -> IO ()

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyFloat"
  c_copyFloat :: Ptr CTHShortStorage -> Ptr CTHFloatStorage -> IO ()

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyDouble"
  c_copyDouble :: Ptr CTHShortStorage -> Ptr CTHDoubleStorage -> IO ()

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyHalf"
  c_copyHalf :: Ptr CTHShortStorage -> Ptr CTHHalfStorage -> IO ()

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHShortStorage -> Ptr CShort -> IO ())

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copy"
  p_copy :: FunPtr (Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ())

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHShortStorage -> Ptr CTHByteStorage -> IO ())

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHShortStorage -> Ptr CTHCharStorage -> IO ())

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ())

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHShortStorage -> Ptr CTHIntStorage -> IO ())

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHShortStorage -> Ptr CTHLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHShortStorage -> Ptr CTHFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHShortStorage -> Ptr CTHDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THShortStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHShortStorage -> Ptr CTHHalfStorage -> IO ())