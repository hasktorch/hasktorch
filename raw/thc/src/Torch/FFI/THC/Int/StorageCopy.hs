{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.StorageCopy
  ( c_rawCopy
  , c_copy
  , c_copyByte
  , c_copyChar
  , c_copyShort
  , c_copyInt
  , c_copyLong
  , c_copyFloat
  , c_copyDouble
  , c_copyHalf
  , c_copyCuda
  , c_copyCPU
  , p_rawCopy
  , p_copy
  , p_copyByte
  , p_copyChar
  , p_copyShort
  , p_copyInt
  , p_copyLong
  , p_copyFloat
  , p_copyDouble
  , p_copyHalf
  , p_copyCuda
  , p_copyCPU
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_rawCopy"
  c_rawCopy :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CInt) -> IO (())

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copy"
  c_copy :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyByte"
  c_copyByte :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyChar"
  c_copyChar :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyShort"
  c_copyShort :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyInt"
  c_copyInt :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyLong"
  c_copyLong :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyFloat"
  c_copyFloat :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyDouble"
  c_copyDouble :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyHalf"
  c_copyHalf :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyCuda"
  c_copyCuda :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THIntStorage_copyCPU"
  c_copyCPU :: Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CInt) -> IO (()))

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copy"
  p_copy :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHHalfStorage) -> IO (()))

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THIntStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (()))