{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleStorageCopy (
    c_THDoubleStorage_rawCopy,
    c_THDoubleStorage_copy,
    c_THDoubleStorage_copyByte,
    c_THDoubleStorage_copyChar,
    c_THDoubleStorage_copyShort,
    c_THDoubleStorage_copyInt,
    c_THDoubleStorage_copyLong,
    c_THDoubleStorage_copyFloat,
    c_THDoubleStorage_copyDouble,
    c_THDoubleStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleStorage_rawCopy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_rawCopy"
  c_THDoubleStorage_rawCopy :: Ptr CTHDoubleStorage -> Ptr CDouble -> IO ()

-- |c_THDoubleStorage_copy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copy"
  c_THDoubleStorage_copy :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_copyByte : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyByte"
  c_THDoubleStorage_copyByte :: Ptr CTHDoubleStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THDoubleStorage_copyChar : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyChar"
  c_THDoubleStorage_copyChar :: Ptr CTHDoubleStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THDoubleStorage_copyShort : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyShort"
  c_THDoubleStorage_copyShort :: Ptr CTHDoubleStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THDoubleStorage_copyInt : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyInt"
  c_THDoubleStorage_copyInt :: Ptr CTHDoubleStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THDoubleStorage_copyLong : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyLong"
  c_THDoubleStorage_copyLong :: Ptr CTHDoubleStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleStorage_copyFloat : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyFloat"
  c_THDoubleStorage_copyFloat :: Ptr CTHDoubleStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THDoubleStorage_copyDouble : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyDouble"
  c_THDoubleStorage_copyDouble :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_copyHalf : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyHalf"
  c_THDoubleStorage_copyHalf :: Ptr CTHDoubleStorage -> Ptr CTHHalfStorage -> IO ()