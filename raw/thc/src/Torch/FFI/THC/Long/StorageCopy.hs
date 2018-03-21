{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Long.StorageCopy where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_rawCopy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_rawCopy"
  c_rawCopy :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr CLong -> IO ()

-- | c_copy :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copy"
  c_copy :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO ()

-- | c_copyByte :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyByte"
  c_copyByte :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THByteStorage -> IO ()

-- | c_copyChar :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyChar"
  c_copyChar :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THCharStorage -> IO ()

-- | c_copyShort :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyShort"
  c_copyShort :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THShortStorage -> IO ()

-- | c_copyInt :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyInt"
  c_copyInt :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THIntStorage -> IO ()

-- | c_copyLong :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyLong"
  c_copyLong :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THLongStorage -> IO ()

-- | c_copyFloat :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyFloat"
  c_copyFloat :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THFloatStorage -> IO ()

-- | c_copyDouble :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyDouble"
  c_copyDouble :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THDoubleStorage -> IO ()

-- | c_copyHalf :  state storage src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyHalf"
  c_copyHalf :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THHalfStorage -> IO ()

-- | c_thCopyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THLongStorage_copyCuda"
  c_thCopyCuda :: Ptr C'THCState -> Ptr C'THLongStorage -> Ptr C'THCLongStorage -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyCuda"
  c_copyCuda :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCStorageCopy.h THCudaLongStorage_copyCPU"
  c_copyCPU :: Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THLongStorage -> IO ()

-- | p_rawCopy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_rawCopy"
  p_rawCopy :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr CLong -> IO ())

-- | p_copy : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copy"
  p_copy :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO ())

-- | p_copyByte : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyByte"
  p_copyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THByteStorage -> IO ())

-- | p_copyChar : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyChar"
  p_copyChar :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THCharStorage -> IO ())

-- | p_copyShort : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyShort"
  p_copyShort :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THShortStorage -> IO ())

-- | p_copyInt : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyInt"
  p_copyInt :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THIntStorage -> IO ())

-- | p_copyLong : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyLong"
  p_copyLong :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THLongStorage -> IO ())

-- | p_copyFloat : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyFloat"
  p_copyFloat :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THFloatStorage -> IO ())

-- | p_copyDouble : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyDouble"
  p_copyDouble :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THDoubleStorage -> IO ())

-- | p_copyHalf : Pointer to function : state storage src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyHalf"
  p_copyHalf :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THHalfStorage -> IO ())

-- | p_thCopyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THLongStorage_copyCuda"
  p_thCopyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THLongStorage -> Ptr C'THCLongStorage -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyCuda"
  p_copyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THCLongStorage -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCStorageCopy.h &THCudaLongStorage_copyCPU"
  p_copyCPU :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> Ptr C'THLongStorage -> IO ())