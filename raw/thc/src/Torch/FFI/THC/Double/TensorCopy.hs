{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorCopy
  ( c_copy
  , c_copyIgnoringOverlaps
  , c_copyByte
  , c_copyChar
  , c_copyShort
  , c_copyInt
  , c_copyLong
  , c_copyFloat
  , c_copyDouble
  , c_copyHalf
  , c_copyCudaByte
  , c_copyCudaChar
  , c_copyCudaShort
  , c_copyCudaInt
  , c_copyCudaLong
  , c_copyCudaDouble
  , c_copyCudaHalf
  , c_copyCuda
  , c_copyCPU
  , c_copyAsyncCPU
  , c_copyAsyncCuda
  , p_copy
  , p_copyIgnoringOverlaps
  , p_copyByte
  , p_copyChar
  , p_copyShort
  , p_copyInt
  , p_copyLong
  , p_copyFloat
  , p_copyDouble
  , p_copyHalf
  , p_copyCudaByte
  , p_copyCudaChar
  , p_copyCudaShort
  , p_copyCudaInt
  , p_copyCudaLong
  , p_copyCudaDouble
  , p_copyCudaHalf
  , p_copyCuda
  , p_copyCPU
  , p_copyAsyncCPU
  , p_copyAsyncCuda
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_copy :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copy"
  c_copy :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_copyIgnoringOverlaps :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyIgnoringOverlaps"
  c_copyIgnoringOverlaps :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_copyByte :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyByte"
  c_copyByte :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | c_copyChar :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyChar"
  c_copyChar :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | c_copyShort :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyShort"
  c_copyShort :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaShortTensor -> IO ()

-- | c_copyInt :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyInt"
  c_copyInt :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaIntTensor -> IO ()

-- | c_copyLong :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyLong"
  c_copyLong :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_copyFloat :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyFloat"
  c_copyFloat :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_copyDouble :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyDouble"
  c_copyDouble :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_copyHalf :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyHalf"
  c_copyHalf :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_copyCudaByte :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCudaByte"
  c_copyCudaByte :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> IO ()

-- | c_copyCudaChar :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCudaChar"
  c_copyCudaChar :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | c_copyCudaShort :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCudaShort"
  c_copyCudaShort :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaShortTensor -> IO ()

-- | c_copyCudaInt :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCudaInt"
  c_copyCudaInt :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaIntTensor -> IO ()

-- | c_copyCudaLong :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCudaLong"
  c_copyCudaLong :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_copyCudaDouble :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCudaDouble"
  c_copyCudaDouble :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_copyCudaHalf :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCudaHalf"
  c_copyCudaHalf :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCuda"
  c_copyCuda :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyCPU"
  c_copyCPU :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_copyAsyncCPU :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyAsyncCPU"
  c_copyAsyncCPU :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_copyAsyncCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THCDoubleTensor_copyAsyncCuda"
  c_copyAsyncCuda :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | p_copy : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copy"
  p_copy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_copyIgnoringOverlaps : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyIgnoringOverlaps"
  p_copyIgnoringOverlaps :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_copyByte : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyByte"
  p_copyByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> IO ())

-- | p_copyChar : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyChar"
  p_copyChar :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaCharTensor -> IO ())

-- | p_copyShort : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyShort"
  p_copyShort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaShortTensor -> IO ())

-- | p_copyInt : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyInt"
  p_copyInt :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaIntTensor -> IO ())

-- | p_copyLong : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyLong"
  p_copyLong :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_copyFloat : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_copyDouble : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_copyHalf : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_copyCudaByte : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCudaByte"
  p_copyCudaByte :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaByteTensor -> IO ())

-- | p_copyCudaChar : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCudaChar"
  p_copyCudaChar :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaCharTensor -> IO ())

-- | p_copyCudaShort : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCudaShort"
  p_copyCudaShort :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaShortTensor -> IO ())

-- | p_copyCudaInt : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCudaInt"
  p_copyCudaInt :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaIntTensor -> IO ())

-- | p_copyCudaLong : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCudaLong"
  p_copyCudaLong :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_copyCudaDouble : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCudaDouble"
  p_copyCudaDouble :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_copyCudaHalf : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCudaHalf"
  p_copyCudaHalf :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCuda"
  p_copyCuda :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyCPU"
  p_copyCPU :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_copyAsyncCPU : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyAsyncCPU"
  p_copyAsyncCPU :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_copyAsyncCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCDoubleTensor_copyAsyncCuda"
  p_copyAsyncCuda :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())