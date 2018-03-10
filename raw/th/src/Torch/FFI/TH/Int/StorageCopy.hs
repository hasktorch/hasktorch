{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_rawCopy"
  c_rawCopy :: Ptr CTHIntStorage -> Ptr CInt -> IO ()

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copy"
  c_copy :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyByte"
  c_copyByte :: Ptr CTHIntStorage -> Ptr CTHByteStorage -> IO ()

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyChar"
  c_copyChar :: Ptr CTHIntStorage -> Ptr CTHCharStorage -> IO ()

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyShort"
  c_copyShort :: Ptr CTHIntStorage -> Ptr CTHShortStorage -> IO ()

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyInt"
  c_copyInt :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyLong"
  c_copyLong :: Ptr CTHIntStorage -> Ptr CTHLongStorage -> IO ()

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyFloat"
  c_copyFloat :: Ptr CTHIntStorage -> Ptr CTHFloatStorage -> IO ()

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyDouble"
  c_copyDouble :: Ptr CTHIntStorage -> Ptr CTHDoubleStorage -> IO ()

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h THIntStorage_copyHalf"
  c_copyHalf :: Ptr CTHIntStorage -> Ptr CTHHalfStorage -> IO ()

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr CTHIntStorage -> Ptr CInt -> IO ())

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copy"
  p_copy :: FunPtr (Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ())

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyByte"
  p_copyByte :: FunPtr (Ptr CTHIntStorage -> Ptr CTHByteStorage -> IO ())

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyChar"
  p_copyChar :: FunPtr (Ptr CTHIntStorage -> Ptr CTHCharStorage -> IO ())

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyShort"
  p_copyShort :: FunPtr (Ptr CTHIntStorage -> Ptr CTHShortStorage -> IO ())

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyInt"
  p_copyInt :: FunPtr (Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ())

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyLong"
  p_copyLong :: FunPtr (Ptr CTHIntStorage -> Ptr CTHLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHIntStorage -> Ptr CTHFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHIntStorage -> Ptr CTHDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THIntStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHIntStorage -> Ptr CTHHalfStorage -> IO ())