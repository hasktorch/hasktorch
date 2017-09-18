{-# LANGUAGE ForeignFunctionInterface#-}

module THHalfTensorCopy (
    c_THHalfTensor_copy,
    c_THHalfTensor_copyByte,
    c_THHalfTensor_copyChar,
    c_THHalfTensor_copyShort,
    c_THHalfTensor_copyInt,
    c_THHalfTensor_copyLong,
    c_THHalfTensor_copyFloat,
    c_THHalfTensor_copyDouble,
    c_THHalfTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfTensor_copy : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copy"
  c_THHalfTensor_copy :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_copyByte : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copyByte"
  c_THHalfTensor_copyByte :: (Ptr CTHHalfTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THHalfTensor_copyChar : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copyChar"
  c_THHalfTensor_copyChar :: (Ptr CTHHalfTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THHalfTensor_copyShort : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copyShort"
  c_THHalfTensor_copyShort :: (Ptr CTHHalfTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THHalfTensor_copyInt : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copyInt"
  c_THHalfTensor_copyInt :: (Ptr CTHHalfTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THHalfTensor_copyLong : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copyLong"
  c_THHalfTensor_copyLong :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THHalfTensor_copyFloat : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copyFloat"
  c_THHalfTensor_copyFloat :: (Ptr CTHHalfTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THHalfTensor_copyDouble : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copyDouble"
  c_THHalfTensor_copyDouble :: (Ptr CTHHalfTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THHalfTensor_copyHalf : tensor src -> void
foreign import ccall "THTensorCopy.h THHalfTensor_copyHalf"
  c_THHalfTensor_copyHalf :: (Ptr CTHHalfTensor) -> Ptr CTHHalfTensor -> IO ()