{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_rawCopy"
  c_rawCopy :: Ptr CTHByteStorage -> Ptr CUChar -> IO ()

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copy"
  c_copy :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyByte"
  c_copyByte :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyChar"
  c_copyChar :: Ptr CTHByteStorage -> Ptr CTHCharStorage -> IO ()

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyShort"
  c_copyShort :: Ptr CTHByteStorage -> Ptr CTHShortStorage -> IO ()

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyInt"
  c_copyInt :: Ptr CTHByteStorage -> Ptr CTHIntStorage -> IO ()

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyLong"
  c_copyLong :: Ptr CTHByteStorage -> Ptr CTHLongStorage -> IO ()

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyFloat"
  c_copyFloat :: Ptr CTHByteStorage -> Ptr CTHFloatStorage -> IO ()

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyDouble"
  c_copyDouble :: Ptr CTHByteStorage -> Ptr CTHDoubleStorage -> IO ()

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyHalf"
  c_copyHalf :: Ptr CTHByteStorage -> Ptr CTHHalfStorage -> IO ()

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHByteStorage -> Ptr CUChar -> IO ())

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copy"
  p_copy :: FunPtr (Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ())

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ())

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHByteStorage -> Ptr CTHCharStorage -> IO ())

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHByteStorage -> Ptr CTHShortStorage -> IO ())

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHByteStorage -> Ptr CTHIntStorage -> IO ())

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHByteStorage -> Ptr CTHLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHByteStorage -> Ptr CTHFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHByteStorage -> Ptr CTHDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHByteStorage -> Ptr CTHHalfStorage -> IO ())