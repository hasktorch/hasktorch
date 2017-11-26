{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleTensorCopy (
    c_THDoubleTensor_copy,
    c_THDoubleTensor_copyByte,
    c_THDoubleTensor_copyChar,
    c_THDoubleTensor_copyShort,
    c_THDoubleTensor_copyInt,
    c_THDoubleTensor_copyLong,
    c_THDoubleTensor_copyFloat,
    c_THDoubleTensor_copyDouble,
    c_THDoubleTensor_copyHalf,
    p_THDoubleTensor_copy,
    p_THDoubleTensor_copyByte,
    p_THDoubleTensor_copyChar,
    p_THDoubleTensor_copyShort,
    p_THDoubleTensor_copyInt,
    p_THDoubleTensor_copyLong,
    p_THDoubleTensor_copyFloat,
    p_THDoubleTensor_copyDouble,
    p_THDoubleTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THDoubleTensor_copy : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copy"
  c_THDoubleTensor_copy :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_copyByte : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyByte"
  c_THDoubleTensor_copyByte :: (Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THDoubleTensor_copyChar : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyChar"
  c_THDoubleTensor_copyChar :: (Ptr CTHDoubleTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THDoubleTensor_copyShort : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyShort"
  c_THDoubleTensor_copyShort :: (Ptr CTHDoubleTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THDoubleTensor_copyInt : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyInt"
  c_THDoubleTensor_copyInt :: (Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THDoubleTensor_copyLong : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyLong"
  c_THDoubleTensor_copyLong :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THDoubleTensor_copyFloat : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyFloat"
  c_THDoubleTensor_copyFloat :: (Ptr CTHDoubleTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THDoubleTensor_copyDouble : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyDouble"
  c_THDoubleTensor_copyDouble :: (Ptr CTHDoubleTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THDoubleTensor_copyHalf : tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyHalf"
  c_THDoubleTensor_copyHalf :: (Ptr CTHDoubleTensor) -> Ptr CTHHalfTensor -> IO ()

-- |p_THDoubleTensor_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copy"
  p_THDoubleTensor_copy :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyByte"
  p_THDoubleTensor_copyByte :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THDoubleTensor_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyChar"
  p_THDoubleTensor_copyChar :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHCharTensor -> IO ())

-- |p_THDoubleTensor_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyShort"
  p_THDoubleTensor_copyShort :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHShortTensor -> IO ())

-- |p_THDoubleTensor_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyInt"
  p_THDoubleTensor_copyInt :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> IO ())

-- |p_THDoubleTensor_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyLong"
  p_THDoubleTensor_copyLong :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> IO ())

-- |p_THDoubleTensor_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyFloat"
  p_THDoubleTensor_copyFloat :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHFloatTensor -> IO ())

-- |p_THDoubleTensor_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyDouble"
  p_THDoubleTensor_copyDouble :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHDoubleTensor -> IO ())

-- |p_THDoubleTensor_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyHalf"
  p_THDoubleTensor_copyHalf :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHHalfTensor -> IO ())