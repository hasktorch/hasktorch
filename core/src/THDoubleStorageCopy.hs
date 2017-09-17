{-# LANGUAGE ForeignFunctionInterface#-}

module THDoubleStorageCopy (
    c_THDoubleStorageCopy_rawCopy,
    c_THDoubleStorageCopy_copy,
    c_THDoubleStorageCopy_copyByte,
    c_THDoubleStorageCopy_copyChar,
    c_THDoubleStorageCopy_copyShort,
    c_THDoubleStorageCopy_copyInt,
    c_THDoubleStorageCopy_copyLong,
    c_THDoubleStorageCopy_copyFloat,
    c_THDoubleStorageCopy_copyDouble,
    c_THDoubleStorageCopy_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleStorageCopy_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_rawCopy"
  c_THDoubleStorageCopy_rawCopy :: Ptr CTHDoubleStorage -> Ptr CDouble -> IO ()

-- |c_THDoubleStorageCopy_copy : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copy"
  c_THDoubleStorageCopy_copy :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorageCopy_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copyByte"
  c_THDoubleStorageCopy_copyByte :: Ptr CTHDoubleStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THDoubleStorageCopy_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copyChar"
  c_THDoubleStorageCopy_copyChar :: Ptr CTHDoubleStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THDoubleStorageCopy_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copyShort"
  c_THDoubleStorageCopy_copyShort :: Ptr CTHDoubleStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THDoubleStorageCopy_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copyInt"
  c_THDoubleStorageCopy_copyInt :: Ptr CTHDoubleStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THDoubleStorageCopy_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copyLong"
  c_THDoubleStorageCopy_copyLong :: Ptr CTHDoubleStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleStorageCopy_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copyFloat"
  c_THDoubleStorageCopy_copyFloat :: Ptr CTHDoubleStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THDoubleStorageCopy_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copyDouble"
  c_THDoubleStorageCopy_copyDouble :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorageCopy_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorageCopy_copyHalf"
  c_THDoubleStorageCopy_copyHalf :: Ptr CTHDoubleStorage -> Ptr CTHHalfStorage -> IO ()