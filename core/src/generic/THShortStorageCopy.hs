{-# LANGUAGE ForeignFunctionInterface #-}

module THShortStorageCopy (
    c_THShortStorage_rawCopy,
    c_THShortStorage_copy,
    c_THShortStorage_copyByte,
    c_THShortStorage_copyChar,
    c_THShortStorage_copyShort,
    c_THShortStorage_copyInt,
    c_THShortStorage_copyLong,
    c_THShortStorage_copyFloat,
    c_THShortStorage_copyDouble,
    c_THShortStorage_copyHalf,
    p_THShortStorage_rawCopy,
    p_THShortStorage_copy,
    p_THShortStorage_copyByte,
    p_THShortStorage_copyChar,
    p_THShortStorage_copyShort,
    p_THShortStorage_copyInt,
    p_THShortStorage_copyLong,
    p_THShortStorage_copyFloat,
    p_THShortStorage_copyDouble,
    p_THShortStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortStorage_rawCopy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_rawCopy"
  c_THShortStorage_rawCopy :: Ptr CTHShortStorage -> Ptr CShort -> IO ()

-- |c_THShortStorage_copy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copy"
  c_THShortStorage_copy :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_copyByte : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copyByte"
  c_THShortStorage_copyByte :: Ptr CTHShortStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THShortStorage_copyChar : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copyChar"
  c_THShortStorage_copyChar :: Ptr CTHShortStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THShortStorage_copyShort : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copyShort"
  c_THShortStorage_copyShort :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_copyInt : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copyInt"
  c_THShortStorage_copyInt :: Ptr CTHShortStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THShortStorage_copyLong : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copyLong"
  c_THShortStorage_copyLong :: Ptr CTHShortStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THShortStorage_copyFloat : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copyFloat"
  c_THShortStorage_copyFloat :: Ptr CTHShortStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THShortStorage_copyDouble : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copyDouble"
  c_THShortStorage_copyDouble :: Ptr CTHShortStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THShortStorage_copyHalf : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THShortStorage_copyHalf"
  c_THShortStorage_copyHalf :: Ptr CTHShortStorage -> Ptr CTHHalfStorage -> IO ()

-- |p_THShortStorage_rawCopy : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_rawCopy"
  p_THShortStorage_rawCopy :: FunPtr (Ptr CTHShortStorage -> Ptr CShort -> IO ())

-- |p_THShortStorage_copy : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copy"
  p_THShortStorage_copy :: FunPtr (Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ())

-- |p_THShortStorage_copyByte : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copyByte"
  p_THShortStorage_copyByte :: FunPtr (Ptr CTHShortStorage -> Ptr CTHByteStorage -> IO ())

-- |p_THShortStorage_copyChar : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copyChar"
  p_THShortStorage_copyChar :: FunPtr (Ptr CTHShortStorage -> Ptr CTHCharStorage -> IO ())

-- |p_THShortStorage_copyShort : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copyShort"
  p_THShortStorage_copyShort :: FunPtr (Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ())

-- |p_THShortStorage_copyInt : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copyInt"
  p_THShortStorage_copyInt :: FunPtr (Ptr CTHShortStorage -> Ptr CTHIntStorage -> IO ())

-- |p_THShortStorage_copyLong : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copyLong"
  p_THShortStorage_copyLong :: FunPtr (Ptr CTHShortStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THShortStorage_copyFloat : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copyFloat"
  p_THShortStorage_copyFloat :: FunPtr (Ptr CTHShortStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_THShortStorage_copyDouble : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copyDouble"
  p_THShortStorage_copyDouble :: FunPtr (Ptr CTHShortStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_THShortStorage_copyHalf : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THShortStorage_copyHalf"
  p_THShortStorage_copyHalf :: FunPtr (Ptr CTHShortStorage -> Ptr CTHHalfStorage -> IO ())