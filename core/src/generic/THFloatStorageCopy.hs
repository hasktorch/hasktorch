{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatStorageCopy (
    c_THFloatStorage_rawCopy,
    c_THFloatStorage_copy,
    c_THFloatStorage_copyByte,
    c_THFloatStorage_copyChar,
    c_THFloatStorage_copyShort,
    c_THFloatStorage_copyInt,
    c_THFloatStorage_copyLong,
    c_THFloatStorage_copyFloat,
    c_THFloatStorage_copyDouble,
    c_THFloatStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatStorage_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_rawCopy"
  c_THFloatStorage_rawCopy :: Ptr CTHFloatStorage -> Ptr CFloat -> IO ()

-- |c_THFloatStorage_copy : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copy"
  c_THFloatStorage_copy :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copyByte"
  c_THFloatStorage_copyByte :: Ptr CTHFloatStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THFloatStorage_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copyChar"
  c_THFloatStorage_copyChar :: Ptr CTHFloatStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THFloatStorage_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copyShort"
  c_THFloatStorage_copyShort :: Ptr CTHFloatStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THFloatStorage_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copyInt"
  c_THFloatStorage_copyInt :: Ptr CTHFloatStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THFloatStorage_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copyLong"
  c_THFloatStorage_copyLong :: Ptr CTHFloatStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatStorage_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copyFloat"
  c_THFloatStorage_copyFloat :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copyDouble"
  c_THFloatStorage_copyDouble :: Ptr CTHFloatStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THFloatStorage_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorage_copyHalf"
  c_THFloatStorage_copyHalf :: Ptr CTHFloatStorage -> Ptr CTHHalfStorage -> IO ()