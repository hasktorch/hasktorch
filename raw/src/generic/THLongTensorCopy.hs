{-# LANGUAGE ForeignFunctionInterface #-}

module THLongTensorCopy (
    c_THLongTensor_copy,
    c_THLongTensor_copyByte,
    c_THLongTensor_copyChar,
    c_THLongTensor_copyShort,
    c_THLongTensor_copyInt,
    c_THLongTensor_copyLong,
    c_THLongTensor_copyFloat,
    c_THLongTensor_copyDouble,
    c_THLongTensor_copyHalf,
    p_THLongTensor_copy,
    p_THLongTensor_copyByte,
    p_THLongTensor_copyChar,
    p_THLongTensor_copyShort,
    p_THLongTensor_copyInt,
    p_THLongTensor_copyLong,
    p_THLongTensor_copyFloat,
    p_THLongTensor_copyDouble,
    p_THLongTensor_copyHalf) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_copy : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copy"
  c_THLongTensor_copy :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_copyByte : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copyByte"
  c_THLongTensor_copyByte :: (Ptr CTHLongTensor) -> Ptr CTHByteTensor -> IO ()

-- |c_THLongTensor_copyChar : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copyChar"
  c_THLongTensor_copyChar :: (Ptr CTHLongTensor) -> Ptr CTHCharTensor -> IO ()

-- |c_THLongTensor_copyShort : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copyShort"
  c_THLongTensor_copyShort :: (Ptr CTHLongTensor) -> Ptr CTHShortTensor -> IO ()

-- |c_THLongTensor_copyInt : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copyInt"
  c_THLongTensor_copyInt :: (Ptr CTHLongTensor) -> Ptr CTHIntTensor -> IO ()

-- |c_THLongTensor_copyLong : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copyLong"
  c_THLongTensor_copyLong :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> IO ()

-- |c_THLongTensor_copyFloat : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copyFloat"
  c_THLongTensor_copyFloat :: (Ptr CTHLongTensor) -> Ptr CTHFloatTensor -> IO ()

-- |c_THLongTensor_copyDouble : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copyDouble"
  c_THLongTensor_copyDouble :: (Ptr CTHLongTensor) -> Ptr CTHDoubleTensor -> IO ()

-- |c_THLongTensor_copyHalf : tensor src -> void
foreign import ccall unsafe "THTensorCopy.h THLongTensor_copyHalf"
  c_THLongTensor_copyHalf :: (Ptr CTHLongTensor) -> Ptr CTHHalfTensor -> IO ()

-- |p_THLongTensor_copy : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copy"
  p_THLongTensor_copy :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_copyByte : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copyByte"
  p_THLongTensor_copyByte :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHByteTensor -> IO ())

-- |p_THLongTensor_copyChar : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copyChar"
  p_THLongTensor_copyChar :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHCharTensor -> IO ())

-- |p_THLongTensor_copyShort : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copyShort"
  p_THLongTensor_copyShort :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHShortTensor -> IO ())

-- |p_THLongTensor_copyInt : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copyInt"
  p_THLongTensor_copyInt :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHIntTensor -> IO ())

-- |p_THLongTensor_copyLong : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copyLong"
  p_THLongTensor_copyLong :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHLongTensor -> IO ())

-- |p_THLongTensor_copyFloat : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copyFloat"
  p_THLongTensor_copyFloat :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHFloatTensor -> IO ())

-- |p_THLongTensor_copyDouble : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copyDouble"
  p_THLongTensor_copyDouble :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHDoubleTensor -> IO ())

-- |p_THLongTensor_copyHalf : Pointer to function tensor src -> void
foreign import ccall unsafe "THTensorCopy.h &THLongTensor_copyHalf"
  p_THLongTensor_copyHalf :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHHalfTensor -> IO ())