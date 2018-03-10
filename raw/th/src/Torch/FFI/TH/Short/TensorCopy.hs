{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.TensorCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_copy :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copy"
  c_copy :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ()

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyByte"
  c_copyByte :: Ptr CTHShortTensor -> Ptr CTHByteTensor -> IO ()

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyChar"
  c_copyChar :: Ptr CTHShortTensor -> Ptr CTHCharTensor -> IO ()

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyShort"
  c_copyShort :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ()

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyInt"
  c_copyInt :: Ptr CTHShortTensor -> Ptr CTHIntTensor -> IO ()

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyLong"
  c_copyLong :: Ptr CTHShortTensor -> Ptr CTHLongTensor -> IO ()

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyFloat"
  c_copyFloat :: Ptr CTHShortTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyDouble"
  c_copyDouble :: Ptr CTHShortTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h THShortTensor_copyHalf"
  c_copyHalf :: Ptr CTHShortTensor -> Ptr CTHHalfTensor -> IO ()

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copy"
  p_copy :: FunPtr (Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ())

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyByte"
  p_copyByte :: FunPtr (Ptr CTHShortTensor -> Ptr CTHByteTensor -> IO ())

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyChar"
  p_copyChar :: FunPtr (Ptr CTHShortTensor -> Ptr CTHCharTensor -> IO ())

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyShort"
  p_copyShort :: FunPtr (Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ())

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyInt"
  p_copyInt :: FunPtr (Ptr CTHShortTensor -> Ptr CTHIntTensor -> IO ())

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyLong"
  p_copyLong :: FunPtr (Ptr CTHShortTensor -> Ptr CTHLongTensor -> IO ())

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHShortTensor -> Ptr CTHFloatTensor -> IO ())

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHShortTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THShortTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHShortTensor -> Ptr CTHHalfTensor -> IO ())