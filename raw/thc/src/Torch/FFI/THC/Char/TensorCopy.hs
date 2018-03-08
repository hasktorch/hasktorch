{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorCopy
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
foreign import ccall "THCTensorCopy.h THCharTensor_copy"
  c_copy :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyIgnoringOverlaps :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyIgnoringOverlaps"
  c_copyIgnoringOverlaps :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyByte :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyByte"
  c_copyByte :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_copyChar :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyChar"
  c_copyChar :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyShort :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyShort"
  c_copyShort :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_copyInt :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyInt"
  c_copyInt :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_copyLong :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyLong"
  c_copyLong :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_copyFloat :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyFloat"
  c_copyFloat :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_copyDouble :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyDouble"
  c_copyDouble :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_copyHalf :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyHalf"
  c_copyHalf :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHHalfTensor) -> IO (())

-- | c_copyCudaByte :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCudaByte"
  c_copyCudaByte :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_copyCudaChar :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCudaChar"
  c_copyCudaChar :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyCudaShort :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCudaShort"
  c_copyCudaShort :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_copyCudaInt :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCudaInt"
  c_copyCudaInt :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_copyCudaLong :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCudaLong"
  c_copyCudaLong :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> IO (())

-- | c_copyCudaDouble :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCudaDouble"
  c_copyCudaDouble :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_copyCudaHalf :  state dst src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCudaHalf"
  c_copyCudaHalf :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHHalfTensor) -> IO (())

-- | c_copyCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCuda"
  c_copyCuda :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyCPU :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyCPU"
  c_copyCPU :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyAsyncCPU :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyAsyncCPU"
  c_copyAsyncCPU :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_copyAsyncCuda :  state self src -> void
foreign import ccall "THCTensorCopy.h THCharTensor_copyAsyncCuda"
  c_copyAsyncCuda :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | p_copy : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copy"
  p_copy :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyIgnoringOverlaps : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyIgnoringOverlaps"
  p_copyIgnoringOverlaps :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyByte : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_copyChar : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyShort : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_copyInt : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_copyLong : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_copyFloat : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_copyDouble : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_copyHalf : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHHalfTensor) -> IO (()))

-- | p_copyCudaByte : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCudaByte"
  p_copyCudaByte :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_copyCudaChar : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCudaChar"
  p_copyCudaChar :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyCudaShort : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCudaShort"
  p_copyCudaShort :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_copyCudaInt : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCudaInt"
  p_copyCudaInt :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_copyCudaLong : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCudaLong"
  p_copyCudaLong :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_copyCudaDouble : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCudaDouble"
  p_copyCudaDouble :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_copyCudaHalf : Pointer to function : state dst src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCudaHalf"
  p_copyCudaHalf :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHHalfTensor) -> IO (()))

-- | p_copyCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCuda"
  p_copyCuda :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyCPU : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyCPU"
  p_copyCPU :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyAsyncCPU : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyAsyncCPU"
  p_copyAsyncCPU :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_copyAsyncCuda : Pointer to function : state self src -> void
foreign import ccall "THCTensorCopy.h &THCharTensor_copyAsyncCuda"
  p_copyAsyncCuda :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))