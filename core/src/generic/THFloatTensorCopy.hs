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
    c_THFloatTensor_copyHalf) where

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