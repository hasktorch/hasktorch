{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.StorageCopy
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
foreign import ccall "THCStorageCopy.h THDoubleStorage_rawCopy"
  c_rawCopy :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CDouble) -> IO (())

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copy"
  c_copy :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyByte"
  c_copyByte :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyChar"
  c_copyChar :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyShort"
  c_copyShort :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyInt"
  c_copyInt :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyLong"
  c_copyLong :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyFloat"
  c_copyFloat :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyDouble"
  c_copyDouble :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyHalf"
  c_copyHalf :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyCuda"
  c_copyCuda :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THDoubleStorage_copyCPU"
  c_copyCPU :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CDouble) -> IO (()))

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copy"
  p_copy :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHHalfStorage) -> IO (()))

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THDoubleStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (()))