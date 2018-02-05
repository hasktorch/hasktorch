{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfStorageCopy
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

-- | c_rawCopy : storage src -> void
foreign import ccall "THStorageCopy.h rawCopy"
  c_rawCopy :: Ptr CTHHalfStorage -> Ptr THHalf -> IO ()

-- | c_copy : storage src -> void
foreign import ccall "THStorageCopy.h copy"
  c_copy :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()

-- | c_copyByte : storage src -> void
foreign import ccall "THStorageCopy.h copyByte"
  c_copyByte :: Ptr CTHHalfStorage -> Ptr CTHByteStorage -> IO ()

-- | c_copyChar : storage src -> void
foreign import ccall "THStorageCopy.h copyChar"
  c_copyChar :: Ptr CTHHalfStorage -> Ptr CTHCharStorage -> IO ()

-- | c_copyShort : storage src -> void
foreign import ccall "THStorageCopy.h copyShort"
  c_copyShort :: Ptr CTHHalfStorage -> Ptr CTHShortStorage -> IO ()

-- | c_copyInt : storage src -> void
foreign import ccall "THStorageCopy.h copyInt"
  c_copyInt :: Ptr CTHHalfStorage -> Ptr CTHIntStorage -> IO ()

-- | c_copyLong : storage src -> void
foreign import ccall "THStorageCopy.h copyLong"
  c_copyLong :: Ptr CTHHalfStorage -> Ptr CTHLongStorage -> IO ()

-- | c_copyFloat : storage src -> void
foreign import ccall "THStorageCopy.h copyFloat"
  c_copyFloat :: Ptr CTHHalfStorage -> Ptr CTHFloatStorage -> IO ()

-- | c_copyDouble : storage src -> void
foreign import ccall "THStorageCopy.h copyDouble"
  c_copyDouble :: Ptr CTHHalfStorage -> Ptr CTHDoubleStorage -> IO ()

-- | c_copyHalf : storage src -> void
foreign import ccall "THStorageCopy.h copyHalf"
  c_copyHalf :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()

-- |p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHHalfStorage -> Ptr THHalf -> IO ())

-- |p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copy"
  p_copy :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ())

-- |p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copyByte"
  p_copyByte :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHByteStorage -> IO ())

-- |p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copyChar"
  p_copyChar :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHCharStorage -> IO ())

-- |p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copyShort"
  p_copyShort :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHShortStorage -> IO ())

-- |p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copyInt"
  p_copyInt :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHIntStorage -> IO ())

-- |p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copyLong"
  p_copyLong :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHLongStorage -> IO ())

-- |p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ())