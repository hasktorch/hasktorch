{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.StorageCopy
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
import THTypes
import Data.Word
import Data.Int

-- | c_rawCopy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_rawCopy"
  c_rawCopy :: Ptr (CTHFloatStorage) -> Ptr (CFloat) -> IO (())

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copy"
  c_copy :: Ptr (CTHFloatStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copyByte"
  c_copyByte :: Ptr (CTHFloatStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copyChar"
  c_copyChar :: Ptr (CTHFloatStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copyShort"
  c_copyShort :: Ptr (CTHFloatStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copyInt"
  c_copyInt :: Ptr (CTHFloatStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copyLong"
  c_copyLong :: Ptr (CTHFloatStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copyFloat"
  c_copyFloat :: Ptr (CTHFloatStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copyDouble"
  c_copyDouble :: Ptr (CTHFloatStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageFloat_copyHalf"
  c_copyHalf :: Ptr (CTHFloatStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_rawCopy"
  p_rawCopy :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CFloat) -> IO (()))

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copy"
  p_copy :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageFloat_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHFloatStorage) -> Ptr (CTHHalfStorage) -> IO (()))