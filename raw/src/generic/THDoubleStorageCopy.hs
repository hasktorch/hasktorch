{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleStorageCopy (
    c_THDoubleStorage_rawCopy,
    c_THDoubleStorage_copy,
    c_THDoubleStorage_copyByte,
    c_THDoubleStorage_copyChar,
    c_THDoubleStorage_copyShort,
    c_THDoubleStorage_copyInt,
    c_THDoubleStorage_copyLong,
    c_THDoubleStorage_copyFloat,
    c_THDoubleStorage_copyDouble,
    c_THDoubleStorage_copyHalf,
    p_THDoubleStorage_rawCopy,
    p_THDoubleStorage_copy,
    p_THDoubleStorage_copyByte,
    p_THDoubleStorage_copyChar,
    p_THDoubleStorage_copyShort,
    p_THDoubleStorage_copyInt,
    p_THDoubleStorage_copyLong,
    p_THDoubleStorage_copyFloat,
    p_THDoubleStorage_copyDouble,
    p_THDoubleStorage_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleStorage_rawCopy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_rawCopy"
  c_THDoubleStorage_rawCopy :: Ptr CTHDoubleStorage -> Ptr CDouble -> IO ()

-- |c_THDoubleStorage_copy : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copy"
  c_THDoubleStorage_copy :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_copyByte : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyByte"
  c_THDoubleStorage_copyByte :: Ptr CTHDoubleStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THDoubleStorage_copyChar : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyChar"
  c_THDoubleStorage_copyChar :: Ptr CTHDoubleStorage -> Ptr CTHCharStorage -> IO ()

-- |c_THDoubleStorage_copyShort : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyShort"
  c_THDoubleStorage_copyShort :: Ptr CTHDoubleStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THDoubleStorage_copyInt : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyInt"
  c_THDoubleStorage_copyInt :: Ptr CTHDoubleStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THDoubleStorage_copyLong : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyLong"
  c_THDoubleStorage_copyLong :: Ptr CTHDoubleStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THDoubleStorage_copyFloat : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyFloat"
  c_THDoubleStorage_copyFloat :: Ptr CTHDoubleStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THDoubleStorage_copyDouble : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyDouble"
  c_THDoubleStorage_copyDouble :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_copyHalf : storage src -> void
foreign import ccall unsafe "THStorageCopy.h THDoubleStorage_copyHalf"
  c_THDoubleStorage_copyHalf :: Ptr CTHDoubleStorage -> Ptr CTHHalfStorage -> IO ()

-- |p_THDoubleStorage_rawCopy : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_rawCopy"
  p_THDoubleStorage_rawCopy :: FunPtr (Ptr CTHDoubleStorage -> Ptr CDouble -> IO ())

-- |p_THDoubleStorage_copy : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copy"
  p_THDoubleStorage_copy :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_THDoubleStorage_copyByte : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copyByte"
  p_THDoubleStorage_copyByte :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHByteStorage -> IO ())

-- |p_THDoubleStorage_copyChar : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copyChar"
  p_THDoubleStorage_copyChar :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHCharStorage -> IO ())

-- |p_THDoubleStorage_copyShort : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copyShort"
  p_THDoubleStorage_copyShort :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHShortStorage -> IO ())

-- |p_THDoubleStorage_copyInt : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copyInt"
  p_THDoubleStorage_copyInt :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHIntStorage -> IO ())

-- |p_THDoubleStorage_copyLong : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copyLong"
  p_THDoubleStorage_copyLong :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THDoubleStorage_copyFloat : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copyFloat"
  p_THDoubleStorage_copyFloat :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_THDoubleStorage_copyDouble : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copyDouble"
  p_THDoubleStorage_copyDouble :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_THDoubleStorage_copyHalf : Pointer to function storage src -> void
foreign import ccall unsafe "THStorageCopy.h &THDoubleStorage_copyHalf"
  p_THDoubleStorage_copyHalf :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHHalfStorage -> IO ())