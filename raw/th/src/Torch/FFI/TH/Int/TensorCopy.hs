{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorCopy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_copy :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copy"
  c_copy_ :: Ptr CTHIntTensor -> Ptr CTHIntTensor -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy = const c_copy_

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyByte"
  c_copyByte_ :: Ptr CTHIntTensor -> Ptr CTHByteTensor -> IO ()

-- | alias of c_copyByte_ with unused argument (for CTHState) to unify backpack signatures.
c_copyByte = const c_copyByte_

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyChar"
  c_copyChar_ :: Ptr CTHIntTensor -> Ptr CTHCharTensor -> IO ()

-- | alias of c_copyChar_ with unused argument (for CTHState) to unify backpack signatures.
c_copyChar = const c_copyChar_

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyShort"
  c_copyShort_ :: Ptr CTHIntTensor -> Ptr CTHShortTensor -> IO ()

-- | alias of c_copyShort_ with unused argument (for CTHState) to unify backpack signatures.
c_copyShort = const c_copyShort_

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyInt"
  c_copyInt_ :: Ptr CTHIntTensor -> Ptr CTHIntTensor -> IO ()

-- | alias of c_copyInt_ with unused argument (for CTHState) to unify backpack signatures.
c_copyInt = const c_copyInt_

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyLong"
  c_copyLong_ :: Ptr CTHIntTensor -> Ptr CTHLongTensor -> IO ()

-- | alias of c_copyLong_ with unused argument (for CTHState) to unify backpack signatures.
c_copyLong = const c_copyLong_

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyFloat"
  c_copyFloat_ :: Ptr CTHIntTensor -> Ptr CTHFloatTensor -> IO ()

-- | alias of c_copyFloat_ with unused argument (for CTHState) to unify backpack signatures.
c_copyFloat = const c_copyFloat_

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyDouble"
  c_copyDouble_ :: Ptr CTHIntTensor -> Ptr CTHDoubleTensor -> IO ()

-- | alias of c_copyDouble_ with unused argument (for CTHState) to unify backpack signatures.
c_copyDouble = const c_copyDouble_

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h THIntTensor_copyHalf"
  c_copyHalf_ :: Ptr CTHIntTensor -> Ptr CTHHalfTensor -> IO ()

-- | alias of c_copyHalf_ with unused argument (for CTHState) to unify backpack signatures.
c_copyHalf = const c_copyHalf_

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copy"
  p_copy_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHIntTensor -> IO ())

-- | alias of p_copy_ with unused argument (for CTHState) to unify backpack signatures.
p_copy = const p_copy_

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyByte"
  p_copyByte_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHByteTensor -> IO ())

-- | alias of p_copyByte_ with unused argument (for CTHState) to unify backpack signatures.
p_copyByte = const p_copyByte_

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyChar"
  p_copyChar_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHCharTensor -> IO ())

-- | alias of p_copyChar_ with unused argument (for CTHState) to unify backpack signatures.
p_copyChar = const p_copyChar_

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyShort"
  p_copyShort_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHShortTensor -> IO ())

-- | alias of p_copyShort_ with unused argument (for CTHState) to unify backpack signatures.
p_copyShort = const p_copyShort_

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyInt"
  p_copyInt_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHIntTensor -> IO ())

-- | alias of p_copyInt_ with unused argument (for CTHState) to unify backpack signatures.
p_copyInt = const p_copyInt_

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyLong"
  p_copyLong_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHLongTensor -> IO ())

-- | alias of p_copyLong_ with unused argument (for CTHState) to unify backpack signatures.
p_copyLong = const p_copyLong_

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyFloat"
  p_copyFloat_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHFloatTensor -> IO ())

-- | alias of p_copyFloat_ with unused argument (for CTHState) to unify backpack signatures.
p_copyFloat = const p_copyFloat_

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyDouble"
  p_copyDouble_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHDoubleTensor -> IO ())

-- | alias of p_copyDouble_ with unused argument (for CTHState) to unify backpack signatures.
p_copyDouble = const p_copyDouble_

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THIntTensor_copyHalf"
  p_copyHalf_ :: FunPtr (Ptr CTHIntTensor -> Ptr CTHHalfTensor -> IO ())

-- | alias of p_copyHalf_ with unused argument (for CTHState) to unify backpack signatures.
p_copyHalf = const p_copyHalf_