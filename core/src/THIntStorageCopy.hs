{-# LANGUAGE ForeignFunctionInterface#-}

module THIntStorageCopy (
    c_THIntStorageCopy_rawCopy,
    c_THIntStorageCopy_copy,
    c_THIntStorageCopy_copyByte,
    c_THIntStorageCopy_copyChar,
    c_THIntStorageCopy_copyShort,
    c_THIntStorageCopy_copyInt,
    c_THIntStorageCopy_copyLong,
    c_THIntStorageCopy_copyFloat,
    c_THIntStorageCopy_copyDouble,
    c_THIntStorageCopy_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntStorageCopy_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_rawCopy"
  c_THIntStorageCopy_rawCopy :: Ptr CTHIntStorage -> Ptr CInt -> IO ()

-- |c_THIntStorageCopy_copy : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copy"
  c_THIntStorageCopy_copy :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THIntStorageCopy_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copyByte"
  c_THIntStorageCopy_copyByte :: Ptr CTHIntStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THIntStorageCopy_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copyChar"
  c_THIntStorageCopy_copyChar :: Ptr CTHIntStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THIntStorageCopy_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copyShort"
  c_THIntStorageCopy_copyShort :: Ptr CTHIntStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THIntStorageCopy_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copyInt"
  c_THIntStorageCopy_copyInt :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THIntStorageCopy_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copyLong"
  c_THIntStorageCopy_copyLong :: Ptr CTHIntStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THIntStorageCopy_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copyFloat"
  c_THIntStorageCopy_copyFloat :: Ptr CTHIntStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THIntStorageCopy_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copyDouble"
  c_THIntStorageCopy_copyDouble :: Ptr CTHIntStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THIntStorageCopy_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorageCopy_copyHalf"
  c_THIntStorageCopy_copyHalf :: Ptr CTHIntStorage -> Ptr CTHHalfStorage -> IO ()