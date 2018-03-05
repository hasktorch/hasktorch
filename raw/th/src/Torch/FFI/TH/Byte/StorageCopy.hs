{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.StorageCopy
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
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_rawCopy"
  c_rawCopy :: Ptr (CTHByteStorage) -> Ptr (CUChar) -> IO (())

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copy"
  c_copy :: Ptr (CTHByteStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copyByte"
  c_copyByte :: Ptr (CTHByteStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copyChar"
  c_copyChar :: Ptr (CTHByteStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copyShort"
  c_copyShort :: Ptr (CTHByteStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copyInt"
  c_copyInt :: Ptr (CTHByteStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copyLong"
  c_copyLong :: Ptr (CTHByteStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copyFloat"
  c_copyFloat :: Ptr (CTHByteStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copyDouble"
  c_copyDouble :: Ptr (CTHByteStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageByte_copyHalf"
  c_copyHalf :: Ptr (CTHByteStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_rawCopy"
  p_rawCopy :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CUChar) -> IO (()))

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copy"
  p_copy :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageByte_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHHalfStorage) -> IO (()))