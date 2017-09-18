{-# LANGUAGE ForeignFunctionInterface#-}

module THIntTensorCopy (
    c_THIntTensor_copy,
    c_THIntTensor_copyByte,
    c_THIntTensor_copyChar,
    c_THIntTensor_copyShort,
    c_THIntTensor_copyInt,
    c_THIntTensor_copyLong,
    c_THIntTensor_copyFloat,
    c_THIntTensor_copyDouble,
    c_THIntTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensor_copy : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copy"
  c_THIntTensor_copy :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_copyByte : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyByte"
  c_THIntTensor_copyByte :: (Ptr CTHIntTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THIntTensor_copyChar : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyChar"
  c_THIntTensor_copyChar :: (Ptr CTHIntTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THIntTensor_copyShort : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyShort"
  c_THIntTensor_copyShort :: (Ptr CTHIntTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THIntTensor_copyInt : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyInt"
  c_THIntTensor_copyInt :: (Ptr CTHIntTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THIntTensor_copyLong : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyLong"
  c_THIntTensor_copyLong :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THIntTensor_copyFloat : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyFloat"
  c_THIntTensor_copyFloat :: (Ptr CTHIntTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THIntTensor_copyDouble : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyDouble"
  c_THIntTensor_copyDouble :: (Ptr CTHIntTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THIntTensor_copyHalf : tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyHalf"
  c_THIntTensor_copyHalf :: (Ptr CTHIntTensor) -> Ptr CTHHalfTensor -> IO ()