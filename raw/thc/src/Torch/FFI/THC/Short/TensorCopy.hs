{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.TensorCopy where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_copy :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copy"
  c_copy :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | c_copyIgnoringOverlaps :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyIgnoringOverlaps"
  c_copyIgnoringOverlaps :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | c_copyByte :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyByte"
  c_copyByte :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THByteTensor -> IO ()

-- | c_copyChar :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyChar"
  c_copyChar :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCharTensor -> IO ()

-- | c_copyShort :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyShort"
  c_copyShort :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_copyInt :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyInt"
  c_copyInt :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THIntTensor -> IO ()

-- | c_copyLong :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyLong"
  c_copyLong :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THLongTensor -> IO ()

-- | c_copyFloat :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyFloat"
  c_copyFloat :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_copyDouble :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyDouble"
  c_copyDouble :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THDoubleTensor -> IO ()

-- | c_copyHalf :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyHalf"
  c_copyHalf :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THHalfTensor -> IO ()

-- | c_copyCudaByte :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyCudaByte"
  c_copyCudaByte :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_copyCudaChar :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyCudaChar"
  c_copyCudaChar :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_copyCudaShort :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyCudaShort"
  c_copyCudaShort :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | c_copyCudaInt :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyCudaInt"
  c_copyCudaInt :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_copyCudaLong :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyCudaLong"
  c_copyCudaLong :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> IO ()

-- | c_copyCudaDouble :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyCudaDouble"
  c_copyCudaDouble :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyCuda"
  c_copyCuda :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | c_thCopyCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THShortTensor_copyCuda"
  c_thCopyCuda :: Ptr C'THCState -> Ptr C'THShortTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyCPU"
  c_copyCPU :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_copyAsyncCPU :  state self src -> void
foreign import ccall "THCTensorCopy.h THCudaShortTensor_copyAsyncCPU"
  c_copyAsyncCPU :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THShortTensor -> IO ()

-- | c_thCopyAsyncCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THShortTensor_copyAsyncCuda"
  c_thCopyAsyncCuda :: Ptr C'THCState -> Ptr C'THShortTensor -> Ptr C'THCudaShortTensor -> IO ()

-- | p_copy : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copy"
  p_copy :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> IO ())

-- | p_copyIgnoringOverlaps : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyIgnoringOverlaps"
  p_copyIgnoringOverlaps :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> IO ())

-- | p_copyByte : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyByte"
  p_copyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THByteTensor -> IO ())

-- | p_copyChar : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyChar"
  p_copyChar :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCharTensor -> IO ())

-- | p_copyShort : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyShort"
  p_copyShort :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_copyInt : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyInt"
  p_copyInt :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THIntTensor -> IO ())

-- | p_copyLong : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyLong"
  p_copyLong :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THLongTensor -> IO ())

-- | p_copyFloat : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_copyDouble : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_copyHalf : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THHalfTensor -> IO ())

-- | p_copyCudaByte : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyCudaByte"
  p_copyCudaByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_copyCudaChar : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyCudaChar"
  p_copyCudaChar :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_copyCudaShort : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyCudaShort"
  p_copyCudaShort :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> IO ())

-- | p_copyCudaInt : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyCudaInt"
  p_copyCudaInt :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_copyCudaLong : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyCudaLong"
  p_copyCudaLong :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> IO ())

-- | p_copyCudaDouble : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyCudaDouble"
  p_copyCudaDouble :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyCuda"
  p_copyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> IO ())

-- | p_thCopyCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THShortTensor_copyCuda"
  p_thCopyCuda :: FunPtr (Ptr C'THCState -> Ptr C'THShortTensor -> Ptr C'THCudaShortTensor -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyCPU"
  p_copyCPU :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_copyAsyncCPU : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCudaShortTensor_copyAsyncCPU"
  p_copyAsyncCPU :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THShortTensor -> IO ())

-- | p_thCopyAsyncCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THShortTensor_copyAsyncCuda"
  p_thCopyAsyncCuda :: FunPtr (Ptr C'THCState -> Ptr C'THShortTensor -> Ptr C'THCudaShortTensor -> IO ())