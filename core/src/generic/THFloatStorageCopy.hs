{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatStorageCopy (
    c_THFloatStorage_rawCopy,
    c_THFloatStorage_copy,
    c_THFloatStorage_copyByte,
    c_THFloatStorage_copyChar,
    c_THFloatStorage_copyShort,
    c_THFloatStorage_copyInt,
    c_THFloatStorage_copyLong,
    c_THFloatStorage_copyFloat,
    c_THFloatStorage_copyDouble,
    c_THFloatStorage_copyHalf,
    p_THFloatStorage_rawCopy,
    p_THFloatStorage_copy,
    p_THFloatStorage_copyByte,
    p_THFloatStorage_copyChar,
    p_THFloatStorage_copyShort,
    p_THFloatStorage_copyInt,
    p_THFloatStorage_copyLong,
    p_THFloatStorage_copyFloat,
    p_THFloatStorage_copyDouble,
    p_THFloatStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatStorage_rawCopy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_rawCopy"
  c_THFloatStorage_rawCopy :: Ptr CTHFloatStorage -> Ptr CFloat -> IO ()

-- |c_THFloatStorage_copy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copy"
  c_THFloatStorage_copy :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_copyByte : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copyByte"
  c_THFloatStorage_copyByte :: Ptr CTHFloatStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THFloatStorage_copyChar : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copyChar"
  c_THFloatStorage_copyChar :: Ptr CTHFloatStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THFloatStorage_copyShort : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copyShort"
  c_THFloatStorage_copyShort :: Ptr CTHFloatStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THFloatStorage_copyInt : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copyInt"
  c_THFloatStorage_copyInt :: Ptr CTHFloatStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THFloatStorage_copyLong : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copyLong"
  c_THFloatStorage_copyLong :: Ptr CTHFloatStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THFloatStorage_copyFloat : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copyFloat"
  c_THFloatStorage_copyFloat :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_copyDouble : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copyDouble"
  c_THFloatStorage_copyDouble :: Ptr CTHFloatStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THFloatStorage_copyHalf : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THFloatStorage_copyHalf"
  c_THFloatStorage_copyHalf :: Ptr CTHFloatStorage -> Ptr CTHHalfStorage -> IO ()

-- |p_THFloatStorage_rawCopy : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_rawCopy"
  p_THFloatStorage_rawCopy :: FunPtr (Ptr CTHFloatStorage -> Ptr CFloat -> IO ())

-- |p_THFloatStorage_copy : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copy"
  p_THFloatStorage_copy :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_THFloatStorage_copyByte : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copyByte"
  p_THFloatStorage_copyByte :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHByteStorage -> IO ())

-- |p_THFloatStorage_copyChar : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copyChar"
  p_THFloatStorage_copyChar :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHCharStorage -> IO ())

-- |p_THFloatStorage_copyShort : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copyShort"
  p_THFloatStorage_copyShort :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHShortStorage -> IO ())

-- |p_THFloatStorage_copyInt : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copyInt"
  p_THFloatStorage_copyInt :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHIntStorage -> IO ())

-- |p_THFloatStorage_copyLong : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copyLong"
  p_THFloatStorage_copyLong :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THFloatStorage_copyFloat : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copyFloat"
  p_THFloatStorage_copyFloat :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_THFloatStorage_copyDouble : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copyDouble"
  p_THFloatStorage_copyDouble :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_THFloatStorage_copyHalf : Pointer to storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THFloatStorage_copyHalf"
  p_THFloatStorage_copyHalf :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHHalfStorage -> IO ())