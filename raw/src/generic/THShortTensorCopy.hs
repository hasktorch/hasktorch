{-# LANGUAGE ForeignFunctionInterface #-}

module THShortTensorCopy (
    c_THShortTensor_copy,
    c_THShortTensor_copyByte,
    c_THShortTensor_copyChar,
    c_THShortTensor_copyShort,
    c_THShortTensor_copyInt,
    c_THShortTensor_copyLong,
    c_THShortTensor_copyFloat,
    c_THShortTensor_copyDouble,
    c_THShortTensor_copyHalf,
    p_THShortTensor_copy,
    p_THShortTensor_copyByte,
    p_THShortTensor_copyChar,
    p_THShortTensor_copyShort,
    p_THShortTensor_copyInt,
    p_THShortTensor_copyLong,
    p_THShortTensor_copyFloat,
    p_THShortTensor_copyDouble,
    p_THShortTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THShortTensor_copy : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copy"
  c_THShortTensor_copy :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_copyByte : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyByte"
  c_THShortTensor_copyByte :: (Ptr CTHShortTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THShortTensor_copyChar : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyChar"
  c_THShortTensor_copyChar :: (Ptr CTHShortTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THShortTensor_copyShort : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyShort"
  c_THShortTensor_copyShort :: (Ptr CTHShortTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THShortTensor_copyInt : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyInt"
  c_THShortTensor_copyInt :: (Ptr CTHShortTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THShortTensor_copyLong : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyLong"
  c_THShortTensor_copyLong :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THShortTensor_copyFloat : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyFloat"
  c_THShortTensor_copyFloat :: (Ptr CTHShortTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THShortTensor_copyDouble : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyDouble"
  c_THShortTensor_copyDouble :: (Ptr CTHShortTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THShortTensor_copyHalf : tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyHalf"
  c_THShortTensor_copyHalf :: (Ptr CTHShortTensor) -> Ptr CTHHalfTensor -> IO ()

-- |p_THShortTensor_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copy"
  p_THShortTensor_copy :: FunPtr ((Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ())

-- |p_THShortTensor_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyByte"
  p_THShortTensor_copyByte :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THShortTensor_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyChar"
  p_THShortTensor_copyChar :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHCharTensor -> IO ())

-- |p_THShortTensor_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyShort"
  p_THShortTensor_copyShort :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHShortTensor -> IO ())

-- |p_THShortTensor_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyInt"
  p_THShortTensor_copyInt :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHIntTensor -> IO ())

-- |p_THShortTensor_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyLong"
  p_THShortTensor_copyLong :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHLongTensor -> IO ())

-- |p_THShortTensor_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyFloat"
  p_THShortTensor_copyFloat :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHFloatTensor -> IO ())

-- |p_THShortTensor_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyDouble"
  p_THShortTensor_copyDouble :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHDoubleTensor -> IO ())

-- |p_THShortTensor_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyHalf"
  p_THShortTensor_copyHalf :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHHalfTensor -> IO ())