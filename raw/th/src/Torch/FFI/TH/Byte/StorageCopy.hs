{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.StorageCopy where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_rawCopy :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_rawCopy"
  c_rawCopy_ :: Ptr C'THByteStorage -> Ptr CUChar -> IO ()

-- | alias of c_rawCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_rawCopy :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr CUChar -> IO ()
c_rawCopy = const c_rawCopy_

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copy"
  c_copy_ :: Ptr C'THByteStorage -> Ptr C'THByteStorage -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THByteStorage -> IO ()
c_copy = const c_copy_

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyByte"
  c_copyByte_ :: Ptr C'THByteStorage -> Ptr C'THByteStorage -> IO ()

-- | alias of c_copyByte_ with unused argument (for CTHState) to unify backpack signatures.
c_copyByte :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THByteStorage -> IO ()
c_copyByte = const c_copyByte_

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyChar"
  c_copyChar_ :: Ptr C'THByteStorage -> Ptr C'THCharStorage -> IO ()

-- | alias of c_copyChar_ with unused argument (for CTHState) to unify backpack signatures.
c_copyChar :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THCharStorage -> IO ()
c_copyChar = const c_copyChar_

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyShort"
  c_copyShort_ :: Ptr C'THByteStorage -> Ptr C'THShortStorage -> IO ()

-- | alias of c_copyShort_ with unused argument (for CTHState) to unify backpack signatures.
c_copyShort :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THShortStorage -> IO ()
c_copyShort = const c_copyShort_

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyInt"
  c_copyInt_ :: Ptr C'THByteStorage -> Ptr C'THIntStorage -> IO ()

-- | alias of c_copyInt_ with unused argument (for CTHState) to unify backpack signatures.
c_copyInt :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THIntStorage -> IO ()
c_copyInt = const c_copyInt_

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyLong"
  c_copyLong_ :: Ptr C'THByteStorage -> Ptr C'THLongStorage -> IO ()

-- | alias of c_copyLong_ with unused argument (for CTHState) to unify backpack signatures.
c_copyLong :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THLongStorage -> IO ()
c_copyLong = const c_copyLong_

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyFloat"
  c_copyFloat_ :: Ptr C'THByteStorage -> Ptr C'THFloatStorage -> IO ()

-- | alias of c_copyFloat_ with unused argument (for CTHState) to unify backpack signatures.
c_copyFloat :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THFloatStorage -> IO ()
c_copyFloat = const c_copyFloat_

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyDouble"
  c_copyDouble_ :: Ptr C'THByteStorage -> Ptr C'THDoubleStorage -> IO ()

-- | alias of c_copyDouble_ with unused argument (for CTHState) to unify backpack signatures.
c_copyDouble :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THDoubleStorage -> IO ()
c_copyDouble = const c_copyDouble_

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h THByteStorage_copyHalf"
  c_copyHalf_ :: Ptr C'THByteStorage -> Ptr C'THHalfStorage -> IO ()

-- | alias of c_copyHalf_ with unused argument (for CTHState) to unify backpack signatures.
c_copyHalf :: Ptr C'THState -> Ptr C'THByteStorage -> Ptr C'THHalfStorage -> IO ()
c_copyHalf = const c_copyHalf_

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr C'THByteStorage -> Ptr CUChar -> IO ())

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copy"
  p_copy :: FunPtr (Ptr C'THByteStorage -> Ptr C'THByteStorage -> IO ())

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyByte"
  p_copyByte :: FunPtr (Ptr C'THByteStorage -> Ptr C'THByteStorage -> IO ())

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyChar"
  p_copyChar :: FunPtr (Ptr C'THByteStorage -> Ptr C'THCharStorage -> IO ())

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyShort"
  p_copyShort :: FunPtr (Ptr C'THByteStorage -> Ptr C'THShortStorage -> IO ())

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyInt"
  p_copyInt :: FunPtr (Ptr C'THByteStorage -> Ptr C'THIntStorage -> IO ())

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyLong"
  p_copyLong :: FunPtr (Ptr C'THByteStorage -> Ptr C'THLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr C'THByteStorage -> Ptr C'THFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr C'THByteStorage -> Ptr C'THDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THByteStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr C'THByteStorage -> Ptr C'THHalfStorage -> IO ())