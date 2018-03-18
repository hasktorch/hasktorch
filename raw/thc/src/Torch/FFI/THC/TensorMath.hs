{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.TensorMath where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_THCudaByteTensor_logicalall :  state self -> int
foreign import ccall "THCTensorMath.h THCudaByteTensor_logicalall"
  c_THCudaByteTensor_logicalall :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> IO CInt

-- | c_THCudaByteTensor_logicalany :  state self -> int
foreign import ccall "THCTensorMath.h THCudaByteTensor_logicalany"
  c_THCudaByteTensor_logicalany :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> IO CInt

-- | p_THCudaByteTensor_logicalall : Pointer to function : state self -> int
foreign import ccall "THCTensorMath.h &THCudaByteTensor_logicalall"
  p_THCudaByteTensor_logicalall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> IO CInt)

-- | p_THCudaByteTensor_logicalany : Pointer to function : state self -> int
foreign import ccall "THCTensorMath.h &THCudaByteTensor_logicalany"
  p_THCudaByteTensor_logicalany :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> IO CInt)