{-# LANGUAGE ForeignFunctionInterface#-}

module THHalfStorageCopy (
    c_THHalfStorageCopy_rawCopy,
    c_THHalfStorageCopy_copy,
    c_THHalfStorageCopy_copyByte,
    c_THHalfStorageCopy_copyChar,
    c_THHalfStorageCopy_copyShort,
    c_THHalfStorageCopy_copyInt,
    c_THHalfStorageCopy_copyLong,
    c_THHalfStorageCopy_copyFloat,
    c_THHalfStorageCopy_copyDouble,
    c_THHalfStorageCopy_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfStorageCopy_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_rawCopy"
  c_THHalfStorageCopy_rawCopy :: Ptr CTHHalfStorage -> Ptr THHalf -> IO ()

-- |c_THHalfStorageCopy_copy : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copy"
  c_THHalfStorageCopy_copy :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorageCopy_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copyByte"
  c_THHalfStorageCopy_copyByte :: Ptr CTHHalfStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THHalfStorageCopy_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copyChar"
  c_THHalfStorageCopy_copyChar :: Ptr CTHHalfStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THHalfStorageCopy_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copyShort"
  c_THHalfStorageCopy_copyShort :: Ptr CTHHalfStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THHalfStorageCopy_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copyInt"
  c_THHalfStorageCopy_copyInt :: Ptr CTHHalfStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THHalfStorageCopy_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copyLong"
  c_THHalfStorageCopy_copyLong :: Ptr CTHHalfStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfStorageCopy_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copyFloat"
  c_THHalfStorageCopy_copyFloat :: Ptr CTHHalfStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THHalfStorageCopy_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copyDouble"
  c_THHalfStorageCopy_copyDouble :: Ptr CTHHalfStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THHalfStorageCopy_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorageCopy_copyHalf"
  c_THHalfStorageCopy_copyHalf :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()