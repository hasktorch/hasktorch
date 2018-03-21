{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorCopy where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_copy :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copy"
  c_copy :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_copyIgnoringOverlaps :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyIgnoringOverlaps"
  c_copyIgnoringOverlaps :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_copyByte :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyByte"
  c_copyByte :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THByteTensor -> IO ()

-- | c_copyChar :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyChar"
  c_copyChar :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCharTensor -> IO ()

-- | c_copyShort :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyShort"
  c_copyShort :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THShortTensor -> IO ()

-- | c_copyInt :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyInt"
  c_copyInt :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THIntTensor -> IO ()

-- | c_copyLong :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyLong"
  c_copyLong :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THLongTensor -> IO ()

-- | c_copyFloat :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyFloat"
  c_copyFloat :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_copyDouble :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyDouble"
  c_copyDouble :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_copyHalf :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyHalf"
  c_copyHalf :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THHalfTensor -> IO ()

-- | c_copyCudaByte :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyCudaByte"
  c_copyCudaByte :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_copyCudaChar :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyCudaChar"
  c_copyCudaChar :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_copyCudaShort :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyCudaShort"
  c_copyCudaShort :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | c_copyCudaInt :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyCudaInt"
  c_copyCudaInt :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_copyCudaLong :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyCudaLong"
  c_copyCudaLong :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaLongTensor -> IO ()

-- | c_copyCudaDouble :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyCudaDouble"
  c_copyCudaDouble :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyCuda"
  c_copyCuda :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_thCopyCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THIntTensor_copyCuda"
  c_thCopyCuda :: Ptr C'THCState -> Ptr C'THIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyCPU"
  c_copyCPU :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THIntTensor -> IO ()

-- | c_copyAsyncCPU :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaIntTensor_copyAsyncCPU"
  c_copyAsyncCPU :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THIntTensor -> IO ()

-- | c_thCopyAsyncCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THIntTensor_copyAsyncCuda"
  c_thCopyAsyncCuda :: Ptr C'THCState -> Ptr C'THIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | p_copy : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copy"
  p_copy :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_copyIgnoringOverlaps : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyIgnoringOverlaps"
  p_copyIgnoringOverlaps :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_copyByte : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyByte"
  p_copyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THByteTensor -> IO ())

-- | p_copyChar : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyChar"
  p_copyChar :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCharTensor -> IO ())

-- | p_copyShort : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyShort"
  p_copyShort :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THShortTensor -> IO ())

-- | p_copyInt : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyInt"
  p_copyInt :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_copyLong : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyLong"
  p_copyLong :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THLongTensor -> IO ())

-- | p_copyFloat : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_copyDouble : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_copyHalf : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THHalfTensor -> IO ())

-- | p_copyCudaByte : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyCudaByte"
  p_copyCudaByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_copyCudaChar : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyCudaChar"
  p_copyCudaChar :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_copyCudaShort : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyCudaShort"
  p_copyCudaShort :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaShortTensor -> IO ())

-- | p_copyCudaInt : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyCudaInt"
  p_copyCudaInt :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_copyCudaLong : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyCudaLong"
  p_copyCudaLong :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaLongTensor -> IO ())

-- | p_copyCudaDouble : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyCudaDouble"
  p_copyCudaDouble :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyCuda"
  p_copyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_thCopyCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THIntTensor_copyCuda"
  p_thCopyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyCPU"
  p_copyCPU :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_copyAsyncCPU : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaIntTensor_copyAsyncCPU"
  p_copyAsyncCPU :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THIntTensor -> IO ())

-- | p_thCopyAsyncCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THIntTensor_copyAsyncCuda"
  p_thCopyAsyncCuda :: FunPtr (Ptr C'THCState -> Ptr C'THIntTensor -> Ptr C'THCudaIntTensor -> IO ())