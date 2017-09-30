{-# LANGUAGE ForeignFunctionInterface #-}

module THByteStorageCopy (
    c_THByteStorage_rawCopy,
    c_THByteStorage_copy,
    c_THByteStorage_copyByte,
    c_THByteStorage_copyChar,
    c_THByteStorage_copyShort,
    c_THByteStorage_copyInt,
    c_THByteStorage_copyLong,
    c_THByteStorage_copyFloat,
    c_THByteStorage_copyDouble,
    c_THByteStorage_copyHalf,
    p_THByteStorage_rawCopy,
    p_THByteStorage_copy,
    p_THByteStorage_copyByte,
    p_THByteStorage_copyChar,
    p_THByteStorage_copyShort,
    p_THByteStorage_copyInt,
    p_THByteStorage_copyLong,
    p_THByteStorage_copyFloat,
    p_THByteStorage_copyDouble,
    p_THByteStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteStorage_rawCopy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_rawCopy"
  c_THByteStorage_rawCopy :: Ptr CTHByteStorage -> Ptr CChar -> IO ()

-- |c_THByteStorage_copy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copy"
  c_THByteStorage_copy :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_copyByte : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copyByte"
  c_THByteStorage_copyByte :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_copyChar : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copyChar"
  c_THByteStorage_copyChar :: Ptr CTHByteStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THByteStorage_copyShort : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copyShort"
  c_THByteStorage_copyShort :: Ptr CTHByteStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THByteStorage_copyInt : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copyInt"
  c_THByteStorage_copyInt :: Ptr CTHByteStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THByteStorage_copyLong : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copyLong"
  c_THByteStorage_copyLong :: Ptr CTHByteStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THByteStorage_copyFloat : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copyFloat"
  c_THByteStorage_copyFloat :: Ptr CTHByteStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THByteStorage_copyDouble : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copyDouble"
  c_THByteStorage_copyDouble :: Ptr CTHByteStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THByteStorage_copyHalf : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THByteStorage_copyHalf"
  c_THByteStorage_copyHalf :: Ptr CTHByteStorage -> Ptr CTHHalfStorage -> IO ()

-- |p_THByteStorage_rawCopy : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_rawCopy"
  p_THByteStorage_rawCopy :: FunPtr (Ptr CTHByteStorage -> Ptr CChar -> IO ())

-- |p_THByteStorage_copy : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copy"
  p_THByteStorage_copy :: FunPtr (Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ())

-- |p_THByteStorage_copyByte : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copyByte"
  p_THByteStorage_copyByte :: FunPtr (Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ())

-- |p_THByteStorage_copyChar : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copyChar"
  p_THByteStorage_copyChar :: FunPtr (Ptr CTHByteStorage -> Ptr CTHCharStorage -> IO ())

-- |p_THByteStorage_copyShort : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copyShort"
  p_THByteStorage_copyShort :: FunPtr (Ptr CTHByteStorage -> Ptr CTHShortStorage -> IO ())

-- |p_THByteStorage_copyInt : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copyInt"
  p_THByteStorage_copyInt :: FunPtr (Ptr CTHByteStorage -> Ptr CTHIntStorage -> IO ())

-- |p_THByteStorage_copyLong : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copyLong"
  p_THByteStorage_copyLong :: FunPtr (Ptr CTHByteStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THByteStorage_copyFloat : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copyFloat"
  p_THByteStorage_copyFloat :: FunPtr (Ptr CTHByteStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_THByteStorage_copyDouble : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copyDouble"
  p_THByteStorage_copyDouble :: FunPtr (Ptr CTHByteStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_THByteStorage_copyHalf : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THByteStorage_copyHalf"
  p_THByteStorage_copyHalf :: FunPtr (Ptr CTHByteStorage -> Ptr CTHHalfStorage -> IO ())