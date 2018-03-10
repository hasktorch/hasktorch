{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_copy :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copy"
  c_copy :: Ptr CTHCharTensor -> Ptr CTHCharTensor -> IO ()

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copyByte"
  c_copyByte :: Ptr CTHCharTensor -> Ptr CTHByteTensor -> IO ()

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copyChar"
  c_copyChar :: Ptr CTHCharTensor -> Ptr CTHCharTensor -> IO ()

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copyShort"
  c_copyShort :: Ptr CTHCharTensor -> Ptr CTHShortTensor -> IO ()

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copyInt"
  c_copyInt :: Ptr CTHCharTensor -> Ptr CTHIntTensor -> IO ()

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copyLong"
  c_copyLong :: Ptr CTHCharTensor -> Ptr CTHLongTensor -> IO ()

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copyFloat"
  c_copyFloat :: Ptr CTHCharTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copyDouble"
  c_copyDouble :: Ptr CTHCharTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h THCharTensor_copyHalf"
  c_copyHalf :: Ptr CTHCharTensor -> Ptr CTHHalfTensor -> IO ()

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copy"
  p_copy :: FunPtr (Ptr CTHCharTensor -> Ptr CTHCharTensor -> IO ())

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copyByte"
  p_copyByte :: FunPtr (Ptr CTHCharTensor -> Ptr CTHByteTensor -> IO ())

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copyChar"
  p_copyChar :: FunPtr (Ptr CTHCharTensor -> Ptr CTHCharTensor -> IO ())

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copyShort"
  p_copyShort :: FunPtr (Ptr CTHCharTensor -> Ptr CTHShortTensor -> IO ())

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copyInt"
  p_copyInt :: FunPtr (Ptr CTHCharTensor -> Ptr CTHIntTensor -> IO ())

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copyLong"
  p_copyLong :: FunPtr (Ptr CTHCharTensor -> Ptr CTHLongTensor -> IO ())

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHCharTensor -> Ptr CTHFloatTensor -> IO ())

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHCharTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THCharTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHCharTensor -> Ptr CTHHalfTensor -> IO ())