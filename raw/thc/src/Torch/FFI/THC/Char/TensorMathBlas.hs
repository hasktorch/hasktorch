{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorMathBlas
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
foreign import ccall "THCTensorMathBlas.h THCharTensor_dot"
  c_dot :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (CLong)

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THCharTensor_addmv"
  c_addmv :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THCharTensor_addmm"
  c_addmm :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THCharTensor_addr"
  c_addr :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCharTensor_addbmm"
  c_addbmm :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCharTensor_baddbmm"
  c_baddbmm :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THCharTensor_dot"
  p_dot :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (CLong))

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THCharTensor_addmv"
  p_addmv :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THCharTensor_addmm"
  p_addmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THCharTensor_addr"
  p_addr :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCharTensor_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCharTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> CChar -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))