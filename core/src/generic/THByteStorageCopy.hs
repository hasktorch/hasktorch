{-# LANGUAGE ForeignFunctionInterface #-}

module THByteStorageCopy (
    c_THByteStorage_rawCopy,
    c_THByteStorage_copy,
    c_THByteStorage_copyByte,
    c_THByteStorage_copyChar,
    c_THByteStorage_copyShort,
    c_THByteStorage_copyInt,
    c_THByteStorage_copyLong,
    c_THByteStorage_copyFloat,
    c_THByteStorage_copyDouble,
    c_THByteStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteStorage_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_rawCopy"
  c_THByteStorage_rawCopy :: Ptr CTHByteStorage -> Ptr CChar -> IO ()

-- |c_THByteStorage_copy : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copy"
  c_THByteStorage_copy :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyByte"
  c_THByteStorage_copyByte :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyChar"
  c_THByteStorage_copyChar :: Ptr CTHByteStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THByteStorage_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyShort"
  c_THByteStorage_copyShort :: Ptr CTHByteStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THByteStorage_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyInt"
  c_THByteStorage_copyInt :: Ptr CTHByteStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THByteStorage_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyLong"
  c_THByteStorage_copyLong :: Ptr CTHByteStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THByteStorage_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyFloat"
  c_THByteStorage_copyFloat :: Ptr CTHByteStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THByteStorage_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyDouble"
  c_THByteStorage_copyDouble :: Ptr CTHByteStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THByteStorage_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyHalf"
  c_THByteStorage_copyHalf :: Ptr CTHByteStorage -> Ptr CTHHalfStorage -> IO ()