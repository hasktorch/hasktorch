{-# LANGUAGE ForeignFunctionInterface#-}

module THLongStorageCopy (
    c_THLongStorageCopy_rawCopy,
    c_THLongStorageCopy_copy,
    c_THLongStorageCopy_copyByte,
    c_THLongStorageCopy_copyChar,
    c_THLongStorageCopy_copyShort,
    c_THLongStorageCopy_copyInt,
    c_THLongStorageCopy_copyLong,
    c_THLongStorageCopy_copyFloat,
    c_THLongStorageCopy_copyDouble,
    c_THLongStorageCopy_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongStorageCopy_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_rawCopy"
  c_THLongStorageCopy_rawCopy :: Ptr CTHLongStorage -> Ptr CLong -> IO ()

-- |c_THLongStorageCopy_copy : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copy"
  c_THLongStorageCopy_copy :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongStorageCopy_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copyByte"
  c_THLongStorageCopy_copyByte :: Ptr CTHLongStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THLongStorageCopy_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copyChar"
  c_THLongStorageCopy_copyChar :: Ptr CTHLongStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THLongStorageCopy_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copyShort"
  c_THLongStorageCopy_copyShort :: Ptr CTHLongStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THLongStorageCopy_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copyInt"
  c_THLongStorageCopy_copyInt :: Ptr CTHLongStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THLongStorageCopy_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copyLong"
  c_THLongStorageCopy_copyLong :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongStorageCopy_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copyFloat"
  c_THLongStorageCopy_copyFloat :: Ptr CTHLongStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THLongStorageCopy_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copyDouble"
  c_THLongStorageCopy_copyDouble :: Ptr CTHLongStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THLongStorageCopy_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THLongStorageCopy_copyHalf"
  c_THLongStorageCopy_copyHalf :: Ptr CTHLongStorage -> Ptr CTHHalfStorage -> IO ()