{-# LANGUAGE ForeignFunctionInterface #-}

module THLongStorageCopy (
    c_THLongStorage_rawCopy,
    c_THLongStorage_copy,
    c_THLongStorage_copyByte,
    c_THLongStorage_copyChar,
    c_THLongStorage_copyShort,
    c_THLongStorage_copyInt,
    c_THLongStorage_copyLong,
    c_THLongStorage_copyFloat,
    c_THLongStorage_copyDouble,
    c_THLongStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongStorage_rawCopy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_rawCopy"
  c_THLongStorage_rawCopy :: Ptr CTHLongStorage -> Ptr CLong -> IO ()

-- |c_THLongStorage_copy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copy"
  c_THLongStorage_copy :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_copyByte : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copyByte"
  c_THLongStorage_copyByte :: Ptr CTHLongStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THLongStorage_copyChar : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copyChar"
  c_THLongStorage_copyChar :: Ptr CTHLongStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THLongStorage_copyShort : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copyShort"
  c_THLongStorage_copyShort :: Ptr CTHLongStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THLongStorage_copyInt : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copyInt"
  c_THLongStorage_copyInt :: Ptr CTHLongStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THLongStorage_copyLong : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copyLong"
  c_THLongStorage_copyLong :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_copyFloat : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copyFloat"
  c_THLongStorage_copyFloat :: Ptr CTHLongStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THLongStorage_copyDouble : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copyDouble"
  c_THLongStorage_copyDouble :: Ptr CTHLongStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THLongStorage_copyHalf : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THLongStorage_copyHalf"
  c_THLongStorage_copyHalf :: Ptr CTHLongStorage -> Ptr CTHHalfStorage -> IO ()