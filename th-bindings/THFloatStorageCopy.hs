{-# LANGUAGE ForeignFunctionInterface#-}

module THFloatStorageCopy (
    c_THFloatStorageCopy_rawCopy,
    c_THFloatStorageCopy_copy,
    c_THFloatStorageCopy_copyByte,
    c_THFloatStorageCopy_copyChar,
    c_THFloatStorageCopy_copyShort,
    c_THFloatStorageCopy_copyInt,
    c_THFloatStorageCopy_copyLong,
    c_THFloatStorageCopy_copyFloat,
    c_THFloatStorageCopy_copyDouble,
    c_THFloatStorageCopy_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatStorageCopy_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_rawCopy"
  c_THFloatStorageCopy_rawCopy :: Ptr CTHFloatStorage -> Ptr CFloat -> IO ()

-- |c_THFloatStorageCopy_copy : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copy"
  c_THFloatStorageCopy_copy :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorageCopy_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copyByte"
  c_THFloatStorageCopy_copyByte :: Ptr CTHFloatStorage -> Ptr CTHFloatByteStorage -> IO ()

-- |c_THFloatStorageCopy_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copyChar"
  c_THFloatStorageCopy_copyChar :: Ptr CTHFloatStorage -> Ptr CTHFloatCharStorage -> IO ()

-- |c_THFloatStorageCopy_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copyShort"
  c_THFloatStorageCopy_copyShort :: Ptr CTHFloatStorage -> Ptr CTHFloatShortStorage -> IO ()

-- |c_THFloatStorageCopy_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copyInt"
  c_THFloatStorageCopy_copyInt :: Ptr CTHFloatStorage -> Ptr CTHFloatIntStorage -> IO ()

-- |c_THFloatStorageCopy_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copyLong"
  c_THFloatStorageCopy_copyLong :: Ptr CTHFloatStorage -> Ptr CTHFloatLongStorage -> IO ()

-- |c_THFloatStorageCopy_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copyFloat"
  c_THFloatStorageCopy_copyFloat :: Ptr CTHFloatStorage -> Ptr CTHFloatFloatStorage -> IO ()

-- |c_THFloatStorageCopy_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copyDouble"
  c_THFloatStorageCopy_copyDouble :: Ptr CTHFloatStorage -> Ptr CTHFloatDoubleStorage -> IO ()

-- |c_THFloatStorageCopy_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THFloatStorageCopy_copyHalf"
  c_THFloatStorageCopy_copyHalf :: Ptr CTHFloatStorage -> Ptr CTHFloatHalfStorage -> IO ()