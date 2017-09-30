{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatTensorCopy (
    c_THFloatTensor_copy,
    c_THFloatTensor_copyByte,
    c_THFloatTensor_copyChar,
    c_THFloatTensor_copyShort,
    c_THFloatTensor_copyInt,
    c_THFloatTensor_copyLong,
    c_THFloatTensor_copyFloat,
    c_THFloatTensor_copyDouble,
    c_THFloatTensor_copyHalf,
    p_THFloatTensor_copy,
    p_THFloatTensor_copyByte,
    p_THFloatTensor_copyChar,
    p_THFloatTensor_copyShort,
    p_THFloatTensor_copyInt,
    p_THFloatTensor_copyLong,
    p_THFloatTensor_copyFloat,
    p_THFloatTensor_copyDouble,
    p_THFloatTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatTensor_copy : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copy"
  c_THFloatTensor_copy :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_copyByte : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copyByte"
  c_THFloatTensor_copyByte :: (Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THFloatTensor_copyChar : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copyChar"
  c_THFloatTensor_copyChar :: (Ptr CTHFloatTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THFloatTensor_copyShort : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copyShort"
  c_THFloatTensor_copyShort :: (Ptr CTHFloatTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THFloatTensor_copyInt : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copyInt"
  c_THFloatTensor_copyInt :: (Ptr CTHFloatTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THFloatTensor_copyLong : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copyLong"
  c_THFloatTensor_copyLong :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THFloatTensor_copyFloat : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copyFloat"
  c_THFloatTensor_copyFloat :: (Ptr CTHFloatTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THFloatTensor_copyDouble : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copyDouble"
  c_THFloatTensor_copyDouble :: (Ptr CTHFloatTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THFloatTensor_copyHalf : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THFloatTensor_copyHalf"
  c_THFloatTensor_copyHalf :: (Ptr CTHFloatTensor) -> Ptr CTHHalfTensor -> IO ()

-- |p_THFloatTensor_copy : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copy"
  p_THFloatTensor_copy :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_copyByte : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copyByte"
  p_THFloatTensor_copyByte :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THFloatTensor_copyChar : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copyChar"
  p_THFloatTensor_copyChar :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHCharTensor -> IO ())

-- |p_THFloatTensor_copyShort : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copyShort"
  p_THFloatTensor_copyShort :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHShortTensor -> IO ())

-- |p_THFloatTensor_copyInt : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copyInt"
  p_THFloatTensor_copyInt :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHIntTensor -> IO ())

-- |p_THFloatTensor_copyLong : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copyLong"
  p_THFloatTensor_copyLong :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> IO ())

-- |p_THFloatTensor_copyFloat : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copyFloat"
  p_THFloatTensor_copyFloat :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHFloatTensor -> IO ())

-- |p_THFloatTensor_copyDouble : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copyDouble"
  p_THFloatTensor_copyDouble :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHDoubleTensor -> IO ())

-- |p_THFloatTensor_copyHalf : Pointer to tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THFloatTensor_copyHalf"
  p_THFloatTensor_copyHalf :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHHalfTensor -> IO ())