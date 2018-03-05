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
foreign import ccall "THStorageCopy.h c_THStorageInt_rawCopy"
  c_rawCopy :: Ptr (CTHIntStorage) -> Ptr (CInt) -> IO (())

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copy"
  c_copy :: Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copyByte"
  c_copyByte :: Ptr (CTHIntStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copyChar"
  c_copyChar :: Ptr (CTHIntStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copyShort"
  c_copyShort :: Ptr (CTHIntStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copyInt"
  c_copyInt :: Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copyLong"
  c_copyLong :: Ptr (CTHIntStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copyFloat"
  c_copyFloat :: Ptr (CTHIntStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copyDouble"
  c_copyDouble :: Ptr (CTHIntStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageInt_copyHalf"
  c_copyHalf :: Ptr (CTHIntStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_rawCopy"
  p_rawCopy :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CInt) -> IO (()))

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copy"
  p_copy :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageInt_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHHalfStorage) -> IO (()))