{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorMathBlas
  ( c_dot
  , c_addmv
  , c_addmm
  , c_addr
  , c_addbmm
  , c_baddbmm
  , p_dot
  , p_addmv
  , p_addmm
  , p_addr
  , p_addbmm
  , p_baddbmm
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_dot :  state self src -> accreal
foreign import ccall "THCTensorMathBlas.h THByteTensor_dot"
  c_dot :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (CLong)

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THByteTensor_addmv"
  c_addmv :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THByteTensor_addmm"
  c_addmm :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THByteTensor_addr"
  c_addr :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THByteTensor_addbmm"
  c_addbmm :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THByteTensor_baddbmm"
  c_baddbmm :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THByteTensor_dot"
  p_dot :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (CLong))

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THByteTensor_addmv"
  p_addmv :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THByteTensor_addmm"
  p_addmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THByteTensor_addr"
  p_addr :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THByteTensor_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THByteTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> CUChar -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))