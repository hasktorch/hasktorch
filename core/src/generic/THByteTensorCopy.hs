{-# LANGUAGE ForeignFunctionInterface #-}

module THByteTensorCopy (
    c_THByteTensor_copy,
    c_THByteTensor_copyByte,
    c_THByteTensor_copyChar,
    c_THByteTensor_copyShort,
    c_THByteTensor_copyInt,
    c_THByteTensor_copyLong,
    c_THByteTensor_copyFloat,
    c_THByteTensor_copyDouble,
    c_THByteTensor_copyHalf,
    p_THByteTensor_copy,
    p_THByteTensor_copyByte,
    p_THByteTensor_copyChar,
    p_THByteTensor_copyShort,
    p_THByteTensor_copyInt,
    p_THByteTensor_copyLong,
    p_THByteTensor_copyFloat,
    p_THByteTensor_copyDouble,
    p_THByteTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensor_copy : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copy"
  c_THByteTensor_copy :: (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_copyByte : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copyByte"
  c_THByteTensor_copyByte :: (Ptr CTHByteTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THByteTensor_copyChar : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copyChar"
  c_THByteTensor_copyChar :: (Ptr CTHByteTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THByteTensor_copyShort : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copyShort"
  c_THByteTensor_copyShort :: (Ptr CTHByteTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THByteTensor_copyInt : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copyInt"
  c_THByteTensor_copyInt :: (Ptr CTHByteTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THByteTensor_copyLong : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copyLong"
  c_THByteTensor_copyLong :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THByteTensor_copyFloat : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copyFloat"
  c_THByteTensor_copyFloat :: (Ptr CTHByteTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THByteTensor_copyDouble : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copyDouble"
  c_THByteTensor_copyDouble :: (Ptr CTHByteTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THByteTensor_copyHalf : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THByteTensor_copyHalf"
  c_THByteTensor_copyHalf :: (Ptr CTHByteTensor) -> Ptr CTHHalfTensor -> IO ()

-- |p_THByteTensor_copy : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copy"
  p_THByteTensor_copy :: FunPtr ((Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ())

-- |p_THByteTensor_copyByte : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copyByte"
  p_THByteTensor_copyByte :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THByteTensor_copyChar : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copyChar"
  p_THByteTensor_copyChar :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHCharTensor -> IO ())

-- |p_THByteTensor_copyShort : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copyShort"
  p_THByteTensor_copyShort :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHShortTensor -> IO ())

-- |p_THByteTensor_copyInt : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copyInt"
  p_THByteTensor_copyInt :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHIntTensor -> IO ())

-- |p_THByteTensor_copyLong : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copyLong"
  p_THByteTensor_copyLong :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHLongTensor -> IO ())

-- |p_THByteTensor_copyFloat : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copyFloat"
  p_THByteTensor_copyFloat :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHFloatTensor -> IO ())

-- |p_THByteTensor_copyDouble : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copyDouble"
  p_THByteTensor_copyDouble :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHDoubleTensor -> IO ())

-- |p_THByteTensor_copyHalf : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THByteTensor_copyHalf"
  p_THByteTensor_copyHalf :: FunPtr ((Ptr CTHByteTensor) -> Ptr CTHHalfTensor -> IO ())