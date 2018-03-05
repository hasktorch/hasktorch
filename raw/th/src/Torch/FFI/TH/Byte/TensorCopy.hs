{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorCopy
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
foreign import ccall "THTensorCopy.h c_THTensorByte_copy"
  c_copy :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorByte_copyByte"
  c_copyByte :: Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorByte_copyChar"
  c_copyChar :: Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorByte_copyShort"
  c_copyShort :: Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorByte_copyInt"
  c_copyInt :: Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorByte_copyLong"
  c_copyLong :: Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorByte_copyFloat"
  c_copyFloat :: Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorByte_copyDouble"
  c_copyDouble :: Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorByte_copyHalf"
  c_copyHalf :: Ptr (CTHByteTensor) -> Ptr (CTHHalfTensor) -> IO (())

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copy"
  p_copy :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorByte_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHHalfTensor) -> IO (()))