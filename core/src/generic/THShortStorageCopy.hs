{-# LANGUAGE ForeignFunctionInterface #-}

module THShortStorageCopy (
    c_THShortStorage_rawCopy,
    c_THShortStorage_copy,
    c_THShortStorage_copyByte,
    c_THShortStorage_copyChar,
    c_THShortStorage_copyShort,
    c_THShortStorage_copyInt,
    c_THShortStorage_copyLong,
    c_THShortStorage_copyFloat,
    c_THShortStorage_copyDouble,
    c_THShortStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortStorage_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_rawCopy"
  c_THShortStorage_rawCopy :: Ptr CTHShortStorage -> Ptr CShort -> IO ()

-- |c_THShortStorage_copy : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copy"
  c_THShortStorage_copy :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyByte"
  c_THShortStorage_copyByte :: Ptr CTHShortStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THShortStorage_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyChar"
  c_THShortStorage_copyChar :: Ptr CTHShortStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THShortStorage_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyShort"
  c_THShortStorage_copyShort :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyInt"
  c_THShortStorage_copyInt :: Ptr CTHShortStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THShortStorage_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyLong"
  c_THShortStorage_copyLong :: Ptr CTHShortStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THShortStorage_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyFloat"
  c_THShortStorage_copyFloat :: Ptr CTHShortStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THShortStorage_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyDouble"
  c_THShortStorage_copyDouble :: Ptr CTHShortStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THShortStorage_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorage_copyHalf"
  c_THShortStorage_copyHalf :: Ptr CTHShortStorage -> Ptr CTHHalfStorage -> IO ()