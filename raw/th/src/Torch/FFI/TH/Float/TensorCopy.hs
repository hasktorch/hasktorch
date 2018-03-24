{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorCopy where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_copy :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copy"
  c_copy_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()
c_copy = const c_copy_

-- | c_copyByte :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copyByte"
  c_copyByte_ :: Ptr C'THFloatTensor -> Ptr C'THByteTensor -> IO ()

-- | alias of c_copyByte_ with unused argument (for CTHState) to unify backpack signatures.
c_copyByte :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THByteTensor -> IO ()
c_copyByte = const c_copyByte_

-- | c_copyChar :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copyChar"
  c_copyChar_ :: Ptr C'THFloatTensor -> Ptr C'THCharTensor -> IO ()

-- | alias of c_copyChar_ with unused argument (for CTHState) to unify backpack signatures.
c_copyChar :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THCharTensor -> IO ()
c_copyChar = const c_copyChar_

-- | c_copyShort :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copyShort"
  c_copyShort_ :: Ptr C'THFloatTensor -> Ptr C'THShortTensor -> IO ()

-- | alias of c_copyShort_ with unused argument (for CTHState) to unify backpack signatures.
c_copyShort :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THShortTensor -> IO ()
c_copyShort = const c_copyShort_

-- | c_copyInt :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copyInt"
  c_copyInt_ :: Ptr C'THFloatTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_copyInt_ with unused argument (for CTHState) to unify backpack signatures.
c_copyInt :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THIntTensor -> IO ()
c_copyInt = const c_copyInt_

-- | c_copyLong :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copyLong"
  c_copyLong_ :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> IO ()

-- | alias of c_copyLong_ with unused argument (for CTHState) to unify backpack signatures.
c_copyLong :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THLongTensor -> IO ()
c_copyLong = const c_copyLong_

-- | c_copyFloat :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copyFloat"
  c_copyFloat_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_copyFloat_ with unused argument (for CTHState) to unify backpack signatures.
c_copyFloat :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()
c_copyFloat = const c_copyFloat_

-- | c_copyDouble :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copyDouble"
  c_copyDouble_ :: Ptr C'THFloatTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_copyDouble_ with unused argument (for CTHState) to unify backpack signatures.
c_copyDouble :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THDoubleTensor -> IO ()
c_copyDouble = const c_copyDouble_

-- | c_copyHalf :  tensor src -> void
foreign import ccall "THTensorCopy.h THFloatTensor_copyHalf"
  c_copyHalf_ :: Ptr C'THFloatTensor -> Ptr C'THHalfTensor -> IO ()

-- | alias of c_copyHalf_ with unused argument (for CTHState) to unify backpack signatures.
c_copyHalf :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THHalfTensor -> IO ()
c_copyHalf = const c_copyHalf_

-- | p_copy : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copy"
  p_copy :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_copyByte : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copyByte"
  p_copyByte :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THByteTensor -> IO ())

-- | p_copyChar : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copyChar"
  p_copyChar :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THCharTensor -> IO ())

-- | p_copyShort : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copyShort"
  p_copyShort :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THShortTensor -> IO ())

-- | p_copyInt : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copyInt"
  p_copyInt :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THIntTensor -> IO ())

-- | p_copyLong : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copyLong"
  p_copyLong :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> IO ())

-- | p_copyFloat : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_copyDouble : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_copyHalf : Pointer to function : tensor src -> void
foreign import ccall "THTensorCopy.h &THFloatTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THHalfTensor -> IO ())