{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorCopy
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
foreign import ccall "THTensorCopy.h c_THTensorChar_copy"
  c_copy :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorChar_copyByte"
  c_copyByte :: Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorChar_copyChar"
  c_copyChar :: Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorChar_copyShort"
  c_copyShort :: Ptr (CTHCharTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorChar_copyInt"
  c_copyInt :: Ptr (CTHCharTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorChar_copyLong"
  c_copyLong :: Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorChar_copyFloat"
  c_copyFloat :: Ptr (CTHCharTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorChar_copyDouble"
  c_copyDouble :: Ptr (CTHCharTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h c_THTensorChar_copyHalf"
  c_copyHalf :: Ptr (CTHCharTensor) -> Ptr (CTHHalfTensor) -> IO (())

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copy"
  p_copy :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &p_THTensorChar_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHCharTensor) -> Ptr (CTHHalfTensor) -> IO (()))