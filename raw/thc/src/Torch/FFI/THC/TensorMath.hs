{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.TensorMath
  ( c_THCudaByteTensor_logicalall
  , c_THCudaByteTensor_logicalany
  , p_THCudaByteTensor_logicalall
  , p_THCudaByteTensor_logicalany
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THCudaByteTensor_logicalall :  state self -> int
foreign import ccall "THCTensorMath.h THCudaByteTensor_logicalall"
  c_THCudaByteTensor_logicalall :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO CInt

-- | c_THCudaByteTensor_logicalany :  state self -> int
foreign import ccall "THCTensorMath.h THCudaByteTensor_logicalany"
  c_THCudaByteTensor_logicalany :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO CInt

-- | p_THCudaByteTensor_logicalall : Pointer to function : state self -> int
foreign import ccall "THCTensorMath.h &THCudaByteTensor_logicalall"
  p_THCudaByteTensor_logicalall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO CInt)

-- | p_THCudaByteTensor_logicalany : Pointer to function : state self -> int
foreign import ccall "THCTensorMath.h &THCudaByteTensor_logicalany"
  p_THCudaByteTensor_logicalany :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO CInt)