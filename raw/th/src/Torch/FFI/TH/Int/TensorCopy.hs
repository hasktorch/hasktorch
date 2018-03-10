{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorCopy
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
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_copy :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copy"
  c_copy :: Ptr CTHIntTensor -> Ptr CTHIntTensor -> IO ()

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyByte"
  c_copyByte :: Ptr CTHIntTensor -> Ptr CTHByteTensor -> IO ()

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyChar"
  c_copyChar :: Ptr CTHIntTensor -> Ptr CTHCharTensor -> IO ()

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyShort"
  c_copyShort :: Ptr CTHIntTensor -> Ptr CTHShortTensor -> IO ()

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyInt"
  c_copyInt :: Ptr CTHIntTensor -> Ptr CTHIntTensor -> IO ()

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyLong"
  c_copyLong :: Ptr CTHIntTensor -> Ptr CTHLongTensor -> IO ()

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyFloat"
  c_copyFloat :: Ptr CTHIntTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyDouble"
  c_copyDouble :: Ptr CTHIntTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyHalf"
  c_copyHalf :: Ptr CTHIntTensor -> Ptr CTHHalfTensor -> IO ()

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copy"
  p_copy :: FunPtr (Ptr CTHIntTensor -> Ptr CTHIntTensor -> IO ())

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyByte"
  p_copyByte :: FunPtr (Ptr CTHIntTensor -> Ptr CTHByteTensor -> IO ())

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyChar"
  p_copyChar :: FunPtr (Ptr CTHIntTensor -> Ptr CTHCharTensor -> IO ())

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyShort"
  p_copyShort :: FunPtr (Ptr CTHIntTensor -> Ptr CTHShortTensor -> IO ())

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyInt"
  p_copyInt :: FunPtr (Ptr CTHIntTensor -> Ptr CTHIntTensor -> IO ())

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyLong"
  p_copyLong :: FunPtr (Ptr CTHIntTensor -> Ptr CTHLongTensor -> IO ())

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHIntTensor -> Ptr CTHFloatTensor -> IO ())

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHIntTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHIntTensor -> Ptr CTHHalfTensor -> IO ())