{-# LANGUAGE ForeignFunctionInterface#-}

module THByteStorageCopy (
    c_THByteStorageCopy_rawCopy,
    c_THByteStorageCopy_copy,
    c_THByteStorageCopy_copyByte,
    c_THByteStorageCopy_copyChar,
    c_THByteStorageCopy_copyShort,
    c_THByteStorageCopy_copyInt,
    c_THByteStorageCopy_copyLong,
    c_THByteStorageCopy_copyFloat,
    c_THByteStorageCopy_copyDouble,
    c_THByteStorageCopy_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteStorageCopy_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_rawCopy"
  c_THByteStorageCopy_rawCopy :: Ptr CTHByteStorage -> Ptr CChar -> IO ()

-- |c_THByteStorageCopy_copy : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copy"
  c_THByteStorageCopy_copy :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THByteStorageCopy_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copyByte"
  c_THByteStorageCopy_copyByte :: Ptr CTHByteStorage -> Ptr CTHByteByteStorage -> IO ()

-- |c_THByteStorageCopy_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copyChar"
  c_THByteStorageCopy_copyChar :: Ptr CTHByteStorage -> Ptr CTHByteCharStorage -> IO ()

-- |c_THByteStorageCopy_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copyShort"
  c_THByteStorageCopy_copyShort :: Ptr CTHByteStorage -> Ptr CTHByteShortStorage -> IO ()

-- |c_THByteStorageCopy_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copyInt"
  c_THByteStorageCopy_copyInt :: Ptr CTHByteStorage -> Ptr CTHByteIntStorage -> IO ()

-- |c_THByteStorageCopy_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copyLong"
  c_THByteStorageCopy_copyLong :: Ptr CTHByteStorage -> Ptr CTHByteLongStorage -> IO ()

-- |c_THByteStorageCopy_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copyFloat"
  c_THByteStorageCopy_copyFloat :: Ptr CTHByteStorage -> Ptr CTHByteFloatStorage -> IO ()

-- |c_THByteStorageCopy_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copyDouble"
  c_THByteStorageCopy_copyDouble :: Ptr CTHByteStorage -> Ptr CTHByteDoubleStorage -> IO ()

-- |c_THByteStorageCopy_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THByteStorageCopy_copyHalf"
  c_THByteStorageCopy_copyHalf :: Ptr CTHByteStorage -> Ptr CTHByteHalfStorage -> IO ()