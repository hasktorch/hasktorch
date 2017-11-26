{-# LANGUAGE ForeignFunctionInterface #-}

module THIntStorageCopy (
    c_THIntStorage_rawCopy,
    c_THIntStorage_copy,
    c_THIntStorage_copyByte,
    c_THIntStorage_copyChar,
    c_THIntStorage_copyShort,
    c_THIntStorage_copyInt,
    c_THIntStorage_copyLong,
    c_THIntStorage_copyFloat,
    c_THIntStorage_copyDouble,
    c_THIntStorage_copyHalf,
    p_THIntStorage_rawCopy,
    p_THIntStorage_copy,
    p_THIntStorage_copyByte,
    p_THIntStorage_copyChar,
    p_THIntStorage_copyShort,
    p_THIntStorage_copyInt,
    p_THIntStorage_copyLong,
    p_THIntStorage_copyFloat,
    p_THIntStorage_copyDouble,
    p_THIntStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THIntStorage_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_rawCopy"
  c_THIntStorage_rawCopy :: Ptr CTHIntStorage -> Ptr CInt -> IO ()

-- |c_THIntStorage_copy : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copy"
  c_THIntStorage_copy :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyByte"
  c_THIntStorage_copyByte :: Ptr CTHIntStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THIntStorage_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyChar"
  c_THIntStorage_copyChar :: Ptr CTHIntStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THIntStorage_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyShort"
  c_THIntStorage_copyShort :: Ptr CTHIntStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THIntStorage_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyInt"
  c_THIntStorage_copyInt :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyLong"
  c_THIntStorage_copyLong :: Ptr CTHIntStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THIntStorage_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyFloat"
  c_THIntStorage_copyFloat :: Ptr CTHIntStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THIntStorage_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyDouble"
  c_THIntStorage_copyDouble :: Ptr CTHIntStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THIntStorage_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyHalf"
  c_THIntStorage_copyHalf :: Ptr CTHIntStorage -> Ptr CTHHalfStorage -> IO ()

-- |p_THIntStorage_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_rawCopy"
  p_THIntStorage_rawCopy :: FunPtr (Ptr CTHIntStorage -> Ptr CInt -> IO ())

-- |p_THIntStorage_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copy"
  p_THIntStorage_copy :: FunPtr (Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ())

-- |p_THIntStorage_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyByte"
  p_THIntStorage_copyByte :: FunPtr (Ptr CTHIntStorage -> Ptr CTHByteStorage -> IO ())

-- |p_THIntStorage_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyChar"
  p_THIntStorage_copyChar :: FunPtr (Ptr CTHIntStorage -> Ptr CTHCharStorage -> IO ())

-- |p_THIntStorage_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyShort"
  p_THIntStorage_copyShort :: FunPtr (Ptr CTHIntStorage -> Ptr CTHShortStorage -> IO ())

-- |p_THIntStorage_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyInt"
  p_THIntStorage_copyInt :: FunPtr (Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ())

-- |p_THIntStorage_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyLong"
  p_THIntStorage_copyLong :: FunPtr (Ptr CTHIntStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THIntStorage_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyFloat"
  p_THIntStorage_copyFloat :: FunPtr (Ptr CTHIntStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_THIntStorage_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyDouble"
  p_THIntStorage_copyDouble :: FunPtr (Ptr CTHIntStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_THIntStorage_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyHalf"
  p_THIntStorage_copyHalf :: FunPtr (Ptr CTHIntStorage -> Ptr CTHHalfStorage -> IO ())