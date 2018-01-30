{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfStorageCopy (
    c_THHalfStorage_rawCopy,
    c_THHalfStorage_copy,
    c_THHalfStorage_copyByte,
    c_THHalfStorage_copyChar,
    c_THHalfStorage_copyShort,
    c_THHalfStorage_copyInt,
    c_THHalfStorage_copyLong,
    c_THHalfStorage_copyFloat,
    c_THHalfStorage_copyDouble,
    c_THHalfStorage_copyHalf,
    p_THHalfStorage_rawCopy,
    p_THHalfStorage_copy,
    p_THHalfStorage_copyByte,
    p_THHalfStorage_copyChar,
    p_THHalfStorage_copyShort,
    p_THHalfStorage_copyInt,
    p_THHalfStorage_copyLong,
    p_THHalfStorage_copyFloat,
    p_THHalfStorage_copyDouble,
    p_THHalfStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THHalfStorage_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_rawCopy"
  c_THHalfStorage_rawCopy :: Ptr CTHHalfStorage -> Ptr CTHHalf -> IO ()

-- |c_THHalfStorage_copy : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copy"
  c_THHalfStorage_copy :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorage_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copyByte"
  c_THHalfStorage_copyByte :: Ptr CTHHalfStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THHalfStorage_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copyChar"
  c_THHalfStorage_copyChar :: Ptr CTHHalfStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THHalfStorage_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copyShort"
  c_THHalfStorage_copyShort :: Ptr CTHHalfStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THHalfStorage_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copyInt"
  c_THHalfStorage_copyInt :: Ptr CTHHalfStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THHalfStorage_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copyLong"
  c_THHalfStorage_copyLong :: Ptr CTHHalfStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THHalfStorage_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copyFloat"
  c_THHalfStorage_copyFloat :: Ptr CTHHalfStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THHalfStorage_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copyDouble"
  c_THHalfStorage_copyDouble :: Ptr CTHHalfStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THHalfStorage_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h THHalfStorage_copyHalf"
  c_THHalfStorage_copyHalf :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()

-- |p_THHalfStorage_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_rawCopy"
  p_THHalfStorage_rawCopy :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHHalf -> IO ())

-- |p_THHalfStorage_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copy"
  p_THHalfStorage_copy :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ())

-- |p_THHalfStorage_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copyByte"
  p_THHalfStorage_copyByte :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHByteStorage -> IO ())

-- |p_THHalfStorage_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copyChar"
  p_THHalfStorage_copyChar :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHCharStorage -> IO ())

-- |p_THHalfStorage_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copyShort"
  p_THHalfStorage_copyShort :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHShortStorage -> IO ())

-- |p_THHalfStorage_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copyInt"
  p_THHalfStorage_copyInt :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHIntStorage -> IO ())

-- |p_THHalfStorage_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copyLong"
  p_THHalfStorage_copyLong :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THHalfStorage_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copyFloat"
  p_THHalfStorage_copyFloat :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_THHalfStorage_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copyDouble"
  p_THHalfStorage_copyDouble :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_THHalfStorage_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THHalfStorage_copyHalf"
  p_THHalfStorage_copyHalf :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ())
