{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfStorageCopy (
    c_THHalfStorage_rawCopy,
    c_THHalfStorage_copy,
    c_THHalfStorage_copyByte,
    c_THHalfStorage_copyChar,
    c_THHalfStorage_copyShort,
    c_THHalfStorage_copyInt,
    c_THHalfStorage_copyLong,
    c_THHalfStorage_copyFloat,
    c_THHalfStorage_copyDouble,
    c_THHalfStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfStorage_rawCopy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_rawCopy"
  c_THHalfStorage_rawCopy :: Ptr CTHHalfStorage -> Ptr THHalf -> IO ()

-- |c_THHalfStorage_copy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copy"
  c_THHalfStorage_copy :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorage_copyByte : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copyByte"
  c_THHalfStorage_copyByte :: Ptr CTHHalfStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THHalfStorage_copyChar : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copyChar"
  c_THHalfStorage_copyChar :: Ptr CTHHalfStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THHalfStorage_copyShort : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copyShort"
  c_THHalfStorage_copyShort :: Ptr CTHHalfStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THHalfStorage_copyInt : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copyInt"
  c_THHalfStorage_copyInt :: Ptr CTHHalfStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THHalfStorage_copyLong : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copyLong"
  c_THHalfStorage_copyLong :: Ptr CTHHalfStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfStorage_copyFloat : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copyFloat"
  c_THHalfStorage_copyFloat :: Ptr CTHHalfStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THHalfStorage_copyDouble : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copyDouble"
  c_THHalfStorage_copyDouble :: Ptr CTHHalfStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THHalfStorage_copyHalf : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THHalfStorage_copyHalf"
  c_THHalfStorage_copyHalf :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()