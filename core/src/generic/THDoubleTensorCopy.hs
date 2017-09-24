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
    c_THDoubleTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

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