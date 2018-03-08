{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorMathBlas
  ( c_dot
  , c_addmv
  , c_addmm
  , c_addr
  , c_addbmm
  , c_baddbmm
  , c_btrifact
  , c_btrisolve
  , p_dot
  , p_addmv
  , p_addmm
  , p_addr
  , p_addbmm
  , p_baddbmm
  , p_btrifact
  , p_btrisolve
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_dot :  state self src -> accreal
foreign import ccall "THCTensorMathBlas.h THDoubleTensor_dot"
  c_dot :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (CDouble)

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THDoubleTensor_addmv"
  c_addmv :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THDoubleTensor_addmm"
  c_addmm :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THDoubleTensor_addr"
  c_addr :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THDoubleTensor_addbmm"
  c_addbmm :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THDoubleTensor_baddbmm"
  c_baddbmm :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_btrifact :  state ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THCTensorMathBlas.h THDoubleTensor_btrifact"
  c_btrifact :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_btrisolve :  state rb_ b atf pivots -> void
foreign import ccall "THCTensorMathBlas.h THDoubleTensor_btrisolve"
  c_btrisolve :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THDoubleTensor_dot"
  p_dot :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (CDouble))

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THDoubleTensor_addmv"
  p_addmv :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THDoubleTensor_addmm"
  p_addmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THDoubleTensor_addr"
  p_addr :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THDoubleTensor_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THDoubleTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_btrifact : Pointer to function : state ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THCTensorMathBlas.h &THDoubleTensor_btrifact"
  p_btrifact :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_btrisolve : Pointer to function : state rb_ b atf pivots -> void
foreign import ccall "THCTensorMathBlas.h &THDoubleTensor_btrisolve"
  p_btrisolve :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> IO (()))