{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.StorageCopy
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
foreign import ccall "THStorageCopy.h c_THStorageChar_rawCopy"
  c_rawCopy :: Ptr (CTHCharStorage) -> Ptr (CChar) -> IO (())

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copy"
  c_copy :: Ptr (CTHCharStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copyByte"
  c_copyByte :: Ptr (CTHCharStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copyChar"
  c_copyChar :: Ptr (CTHCharStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copyShort"
  c_copyShort :: Ptr (CTHCharStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copyInt"
  c_copyInt :: Ptr (CTHCharStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copyLong"
  c_copyLong :: Ptr (CTHCharStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copyFloat"
  c_copyFloat :: Ptr (CTHCharStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copyDouble"
  c_copyDouble :: Ptr (CTHCharStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageChar_copyHalf"
  c_copyHalf :: Ptr (CTHCharStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_rawCopy"
  p_rawCopy :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CChar) -> IO (()))

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copy"
  p_copy :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageChar_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHHalfStorage) -> IO (()))