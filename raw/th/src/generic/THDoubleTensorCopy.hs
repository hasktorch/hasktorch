{-# LANGUAGE ForeignFunctionInterface #-}
module THDoubleTensorCopy
  ( c_copy
  , c_copyByte
  , c_copyChar
  , c_copyShort
  , c_copyInt
  , c_copyLong
  , c_copyFloat
  , c_copyDouble
  , c_copyHalf
  , p_copy
  , p_copyByte
  , p_copyChar
  , p_copyShort
  , p_copyInt
  , p_copyLong
  , p_copyFloat
  , p_copyDouble
  , p_copyHalf
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_copy :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copy"
  c_copy :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyByte"
  c_copyByte :: Ptr CTHDoubleTensor -> Ptr CTHByteTensor -> IO ()

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyChar"
  c_copyChar :: Ptr CTHDoubleTensor -> Ptr CTHCharTensor -> IO ()

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyShort"
  c_copyShort :: Ptr CTHDoubleTensor -> Ptr CTHShortTensor -> IO ()

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyInt"
  c_copyInt :: Ptr CTHDoubleTensor -> Ptr CTHIntTensor -> IO ()

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyLong"
  c_copyLong :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> IO ()

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyFloat"
  c_copyFloat :: Ptr CTHDoubleTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyDouble"
  c_copyDouble :: Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h THDoubleTensor_copyHalf"
  c_copyHalf :: Ptr CTHDoubleTensor -> Ptr CTHHalfTensor -> IO ()

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copy"
  p_copy :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyByte"
  p_copyByte :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHByteTensor -> IO ())

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyChar"
  p_copyChar :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHCharTensor -> IO ())

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyShort"
  p_copyShort :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHShortTensor -> IO ())

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyInt"
  p_copyInt :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHIntTensor -> IO ())

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyLong"
  p_copyLong :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> IO ())

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHFloatTensor -> IO ())

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THDoubleTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHHalfTensor -> IO ())