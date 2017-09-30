{-# LANGUAGE ForeignFunctionInterface #-}

module THIntTensorCopy (
    c_THIntTensor_copy,
    c_THIntTensor_copyByte,
    c_THIntTensor_copyChar,
    c_THIntTensor_copyShort,
    c_THIntTensor_copyInt,
    c_THIntTensor_copyLong,
    c_THIntTensor_copyFloat,
    c_THIntTensor_copyDouble,
    c_THIntTensor_copyHalf,
    p_THIntTensor_copy,
    p_THIntTensor_copyByte,
    p_THIntTensor_copyChar,
    p_THIntTensor_copyShort,
    p_THIntTensor_copyInt,
    p_THIntTensor_copyLong,
    p_THIntTensor_copyFloat,
    p_THIntTensor_copyDouble,
    p_THIntTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensor_copy : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copy"
  c_THIntTensor_copy :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_copyByte : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copyByte"
  c_THIntTensor_copyByte :: (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THIntTensor_copyChar : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copyChar"
  c_THIntTensor_copyChar :: (Ptr CTHIntTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THIntTensor_copyShort : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copyShort"
  c_THIntTensor_copyShort :: (Ptr CTHIntTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THIntTensor_copyInt : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copyInt"
  c_THIntTensor_copyInt :: (Ptr CTHIntTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THIntTensor_copyLong : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copyLong"
  c_THIntTensor_copyLong :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THIntTensor_copyFloat : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copyFloat"
  c_THIntTensor_copyFloat :: (Ptr CTHIntTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THIntTensor_copyDouble : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copyDouble"
  c_THIntTensor_copyDouble :: (Ptr CTHIntTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THIntTensor_copyHalf : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THIntTensor_copyHalf"
  c_THIntTensor_copyHalf :: (Ptr CTHIntTensor) -> Ptr CTHHalfTensor -> IO ()

-- |p_THIntTensor_copy : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copy"
  p_THIntTensor_copy :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_copyByte : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copyByte"
  p_THIntTensor_copyByte :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THIntTensor_copyChar : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copyChar"
  p_THIntTensor_copyChar :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHCharTensor -> IO ())

-- |p_THIntTensor_copyShort : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copyShort"
  p_THIntTensor_copyShort :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHShortTensor -> IO ())

-- |p_THIntTensor_copyInt : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copyInt"
  p_THIntTensor_copyInt :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntTensor -> IO ())

-- |p_THIntTensor_copyLong : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copyLong"
  p_THIntTensor_copyLong :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHLongTensor -> IO ())

-- |p_THIntTensor_copyFloat : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copyFloat"
  p_THIntTensor_copyFloat :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHFloatTensor -> IO ())

-- |p_THIntTensor_copyDouble : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copyDouble"
  p_THIntTensor_copyDouble :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHDoubleTensor -> IO ())

-- |p_THIntTensor_copyHalf : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THIntTensor_copyHalf"
  p_THIntTensor_copyHalf :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHHalfTensor -> IO ())