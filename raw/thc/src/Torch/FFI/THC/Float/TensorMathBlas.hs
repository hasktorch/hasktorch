{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorMathBlas
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
foreign import ccall "THCTensorMathBlas.h THFloatTensor_dot"
  c_dot :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THFloatTensor_addmv"
  c_addmv :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THFloatTensor_addmm"
  c_addmm :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THFloatTensor_addr"
  c_addr :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THFloatTensor_addbmm"
  c_addbmm :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THFloatTensor_baddbmm"
  c_baddbmm :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_btrifact :  state ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THCTensorMathBlas.h THFloatTensor_btrifact"
  c_btrifact :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHFloatTensor) -> IO (())

-- | c_btrisolve :  state rb_ b atf pivots -> void
foreign import ccall "THCTensorMathBlas.h THFloatTensor_btrisolve"
  c_btrisolve :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THFloatTensor_dot"
  p_dot :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THFloatTensor_addmv"
  p_addmv :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THFloatTensor_addmm"
  p_addmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THFloatTensor_addr"
  p_addr :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THFloatTensor_addbmm"
  p_addbmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THFloatTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> CFloat -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_btrifact : Pointer to function : state ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THCTensorMathBlas.h &THFloatTensor_btrifact"
  p_btrifact :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_btrisolve : Pointer to function : state rb_ b atf pivots -> void
foreign import ccall "THCTensorMathBlas.h &THFloatTensor_btrisolve"
  p_btrisolve :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> Ptr (CTHIntTensor) -> IO (()))