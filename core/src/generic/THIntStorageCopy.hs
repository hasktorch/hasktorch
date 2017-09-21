{-# LANGUAGE ForeignFunctionInterface#-}

module THIntStorageCopy (
    c_THIntStorage_rawCopy,
    c_THIntStorage_copy,
    c_THIntStorage_copyByte,
    c_THIntStorage_copyChar,
    c_THIntStorage_copyShort,
    c_THIntStorage_copyInt,
    c_THIntStorage_copyLong,
    c_THIntStorage_copyFloat,
    c_THIntStorage_copyDouble,
    c_THIntStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntStorage_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_rawCopy"
  c_THIntStorage_rawCopy :: Ptr CTHIntStorage -> Ptr CInt -> IO ()

-- |c_THIntStorage_copy : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copy"
  c_THIntStorage_copy :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyByte"
  c_THIntStorage_copyByte :: Ptr CTHIntStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THIntStorage_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyChar"
  c_THIntStorage_copyChar :: Ptr CTHIntStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THIntStorage_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyShort"
  c_THIntStorage_copyShort :: Ptr CTHIntStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THIntStorage_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyInt"
  c_THIntStorage_copyInt :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyLong"
  c_THIntStorage_copyLong :: Ptr CTHIntStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THIntStorage_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyFloat"
  c_THIntStorage_copyFloat :: Ptr CTHIntStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THIntStorage_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyDouble"
  c_THIntStorage_copyDouble :: Ptr CTHIntStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THIntStorage_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyHalf"
  c_THIntStorage_copyHalf :: Ptr CTHIntStorage -> Ptr CTHHalfStorage -> IO ()