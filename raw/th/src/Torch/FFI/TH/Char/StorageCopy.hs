{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.StorageCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_rawCopy :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_rawCopy"
  c_rawCopy_ :: Ptr CTHCharStorage -> Ptr CChar -> IO ()

-- | alias of c_rawCopy_ with unused argument (for CTHState) to unify backpack signatures.
c_rawCopy = const c_rawCopy_

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copy"
  c_copy_ :: Ptr CTHCharStorage -> Ptr CTHCharStorage -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy = const c_copy_

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copyByte"
  c_copyByte_ :: Ptr CTHCharStorage -> Ptr CTHByteStorage -> IO ()

-- | alias of c_copyByte_ with unused argument (for CTHState) to unify backpack signatures.
c_copyByte = const c_copyByte_

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copyChar"
  c_copyChar_ :: Ptr CTHCharStorage -> Ptr CTHCharStorage -> IO ()

-- | alias of c_copyChar_ with unused argument (for CTHState) to unify backpack signatures.
c_copyChar = const c_copyChar_

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copyShort"
  c_copyShort_ :: Ptr CTHCharStorage -> Ptr CTHShortStorage -> IO ()

-- | alias of c_copyShort_ with unused argument (for CTHState) to unify backpack signatures.
c_copyShort = const c_copyShort_

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copyInt"
  c_copyInt_ :: Ptr CTHCharStorage -> Ptr CTHIntStorage -> IO ()

-- | alias of c_copyInt_ with unused argument (for CTHState) to unify backpack signatures.
c_copyInt = const c_copyInt_

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copyLong"
  c_copyLong_ :: Ptr CTHCharStorage -> Ptr CTHLongStorage -> IO ()

-- | alias of c_copyLong_ with unused argument (for CTHState) to unify backpack signatures.
c_copyLong = const c_copyLong_

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copyFloat"
  c_copyFloat_ :: Ptr CTHCharStorage -> Ptr CTHFloatStorage -> IO ()

-- | alias of c_copyFloat_ with unused argument (for CTHState) to unify backpack signatures.
c_copyFloat = const c_copyFloat_

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copyDouble"
  c_copyDouble_ :: Ptr CTHCharStorage -> Ptr CTHDoubleStorage -> IO ()

-- | alias of c_copyDouble_ with unused argument (for CTHState) to unify backpack signatures.
c_copyDouble = const c_copyDouble_

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h THCharStorage_copyHalf"
  c_copyHalf_ :: Ptr CTHCharStorage -> Ptr CTHHalfStorage -> IO ()

-- | alias of c_copyHalf_ with unused argument (for CTHState) to unify backpack signatures.
c_copyHalf = const c_copyHalf_

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_rawCopy"
  p_rawCopy_ :: FunPtr (Ptr CTHCharStorage -> Ptr CChar -> IO ())

-- | alias of p_rawCopy_ with unused argument (for CTHState) to unify backpack signatures.
p_rawCopy = const p_rawCopy_

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copy"
  p_copy_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHCharStorage -> IO ())

-- | alias of p_copy_ with unused argument (for CTHState) to unify backpack signatures.
p_copy = const p_copy_

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copyByte"
  p_copyByte_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHByteStorage -> IO ())

-- | alias of p_copyByte_ with unused argument (for CTHState) to unify backpack signatures.
p_copyByte = const p_copyByte_

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copyChar"
  p_copyChar_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHCharStorage -> IO ())

-- | alias of p_copyChar_ with unused argument (for CTHState) to unify backpack signatures.
p_copyChar = const p_copyChar_

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copyShort"
  p_copyShort_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHShortStorage -> IO ())

-- | alias of p_copyShort_ with unused argument (for CTHState) to unify backpack signatures.
p_copyShort = const p_copyShort_

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copyInt"
  p_copyInt_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHIntStorage -> IO ())

-- | alias of p_copyInt_ with unused argument (for CTHState) to unify backpack signatures.
p_copyInt = const p_copyInt_

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copyLong"
  p_copyLong_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHLongStorage -> IO ())

-- | alias of p_copyLong_ with unused argument (for CTHState) to unify backpack signatures.
p_copyLong = const p_copyLong_

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copyFloat"
  p_copyFloat_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHFloatStorage -> IO ())

-- | alias of p_copyFloat_ with unused argument (for CTHState) to unify backpack signatures.
p_copyFloat = const p_copyFloat_

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copyDouble"
  p_copyDouble_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHDoubleStorage -> IO ())

-- | alias of p_copyDouble_ with unused argument (for CTHState) to unify backpack signatures.
p_copyDouble = const p_copyDouble_

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &THCharStorage_copyHalf"
  p_copyHalf_ :: FunPtr (Ptr CTHCharStorage -> Ptr CTHHalfStorage -> IO ())

-- | alias of p_copyHalf_ with unused argument (for CTHState) to unify backpack signatures.
p_copyHalf = const p_copyHalf_