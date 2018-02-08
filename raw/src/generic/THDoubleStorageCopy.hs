{-# LANGUAGE ForeignFunctionInterface #-}
module THDoubleStorageCopy
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
foreign import ccall "THStorageCopy.h THDoubleStorage_rawCopy"
  c_rawCopy :: Ptr CTHDoubleStorage -> Ptr CDouble -> IO ()

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copy"
  c_copy :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copyByte"
  c_copyByte :: Ptr CTHDoubleStorage -> Ptr CTHByteStorage -> IO ()

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copyChar"
  c_copyChar :: Ptr CTHDoubleStorage -> Ptr CTHCharStorage -> IO ()

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copyShort"
  c_copyShort :: Ptr CTHDoubleStorage -> Ptr CTHShortStorage -> IO ()

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copyInt"
  c_copyInt :: Ptr CTHDoubleStorage -> Ptr CTHIntStorage -> IO ()

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copyLong"
  c_copyLong :: Ptr CTHDoubleStorage -> Ptr CTHLongStorage -> IO ()

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copyFloat"
  c_copyFloat :: Ptr CTHDoubleStorage -> Ptr CTHFloatStorage -> IO ()

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copyDouble"
  c_copyDouble :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h THDoubleStorage_copyHalf"
  c_copyHalf :: Ptr CTHDoubleStorage -> Ptr CTHHalfStorage -> IO ()

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHDoubleStorage -> Ptr CDouble -> IO ())

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copy"
  p_copy :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ())

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHByteStorage -> IO ())

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHCharStorage -> IO ())

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHShortStorage -> IO ())

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHIntStorage -> IO ())

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THDoubleStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHHalfStorage -> IO ())