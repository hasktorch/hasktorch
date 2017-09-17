{-# LANGUAGE ForeignFunctionInterface#-}

module THShortStorageCopy (
    c_THShortStorageCopy_rawCopy,
    c_THShortStorageCopy_copy,
    c_THShortStorageCopy_copyByte,
    c_THShortStorageCopy_copyChar,
    c_THShortStorageCopy_copyShort,
    c_THShortStorageCopy_copyInt,
    c_THShortStorageCopy_copyLong,
    c_THShortStorageCopy_copyFloat,
    c_THShortStorageCopy_copyDouble,
    c_THShortStorageCopy_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortStorageCopy_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_rawCopy"
  c_THShortStorageCopy_rawCopy :: Ptr CTHShortStorage -> Ptr CShort -> IO ()

-- |c_THShortStorageCopy_copy : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copy"
  c_THShortStorageCopy_copy :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorageCopy_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copyByte"
  c_THShortStorageCopy_copyByte :: Ptr CTHShortStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THShortStorageCopy_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copyChar"
  c_THShortStorageCopy_copyChar :: Ptr CTHShortStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THShortStorageCopy_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copyShort"
  c_THShortStorageCopy_copyShort :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorageCopy_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copyInt"
  c_THShortStorageCopy_copyInt :: Ptr CTHShortStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THShortStorageCopy_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copyLong"
  c_THShortStorageCopy_copyLong :: Ptr CTHShortStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THShortStorageCopy_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copyFloat"
  c_THShortStorageCopy_copyFloat :: Ptr CTHShortStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THShortStorageCopy_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copyDouble"
  c_THShortStorageCopy_copyDouble :: Ptr CTHShortStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THShortStorageCopy_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THShortStorageCopy_copyHalf"
  c_THShortStorageCopy_copyHalf :: Ptr CTHShortStorage -> Ptr CTHHalfStorage -> IO ()