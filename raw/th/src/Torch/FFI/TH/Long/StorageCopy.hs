{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.StorageCopy
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
foreign import ccall "THStorageCopy.h c_THStorageLong_rawCopy"
  c_rawCopy :: Ptr (CTHLongStorage) -> Ptr (CLong) -> IO (())

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copy"
  c_copy :: Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copyByte"
  c_copyByte :: Ptr (CTHLongStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copyChar"
  c_copyChar :: Ptr (CTHLongStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copyShort"
  c_copyShort :: Ptr (CTHLongStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copyInt"
  c_copyInt :: Ptr (CTHLongStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copyLong"
  c_copyLong :: Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copyFloat"
  c_copyFloat :: Ptr (CTHLongStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copyDouble"
  c_copyDouble :: Ptr (CTHLongStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageLong_copyHalf"
  c_copyHalf :: Ptr (CTHLongStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_rawCopy"
  p_rawCopy :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CLong) -> IO (()))

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copy"
  p_copy :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageLong_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHLongStorage) -> Ptr (CTHHalfStorage) -> IO (()))