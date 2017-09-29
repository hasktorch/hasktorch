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
    c_THByteTensor_copyHalf) where

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