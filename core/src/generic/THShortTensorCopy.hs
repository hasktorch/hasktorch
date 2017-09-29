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
    c_THShortTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortTensor_copy : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copy"
  c_THShortTensor_copy :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_copyByte : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copyByte"
  c_THShortTensor_copyByte :: (Ptr CTHShortTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THShortTensor_copyChar : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copyChar"
  c_THShortTensor_copyChar :: (Ptr CTHShortTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THShortTensor_copyShort : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copyShort"
  c_THShortTensor_copyShort :: (Ptr CTHShortTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THShortTensor_copyInt : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copyInt"
  c_THShortTensor_copyInt :: (Ptr CTHShortTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THShortTensor_copyLong : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copyLong"
  c_THShortTensor_copyLong :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THShortTensor_copyFloat : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copyFloat"
  c_THShortTensor_copyFloat :: (Ptr CTHShortTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THShortTensor_copyDouble : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copyDouble"
  c_THShortTensor_copyDouble :: (Ptr CTHShortTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THShortTensor_copyHalf : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THShortTensor_copyHalf"
  c_THShortTensor_copyHalf :: (Ptr CTHShortTensor) -> Ptr CTHHalfTensor -> IO ()