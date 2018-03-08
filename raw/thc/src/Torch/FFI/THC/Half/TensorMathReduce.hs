{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.TensorMathReduce
  ( c_sum
  , c_prod
  , c_sumall
  , c_prodall
  , c_min
  , c_max
  , c_minall
  , c_maxall
  , c_medianall
  , c_median
  , p_sum
  , p_prod
  , p_sumall
  , p_prodall
  , p_min
  , p_max
  , p_minall
  , p_maxall
  , p_medianall
  , p_median
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THHalfTensor_sum"
  c_sum :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (())

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THHalfTensor_prod"
  c_prod :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (())

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THHalfTensor_sumall"
  c_sumall :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CFloat)

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THHalfTensor_prodall"
  c_prodall :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CFloat)

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THHalfTensor_min"
  c_min :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (())

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THHalfTensor_max"
  c_max :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (())

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THHalfTensor_minall"
  c_minall :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CTHHalf)

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THHalfTensor_maxall"
  c_maxall :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CTHHalf)

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THHalfTensor_medianall"
  c_medianall :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CTHHalf)

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THHalfTensor_median"
  c_median :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (())

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_sum"
  p_sum :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (()))

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_prod"
  p_prod :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (()))

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_sumall"
  p_sumall :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CFloat))

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_prodall"
  p_prodall :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CFloat))

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_min"
  p_min :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (()))

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_max"
  p_max :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (()))

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_minall"
  p_minall :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CTHHalf))

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_maxall"
  p_maxall :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CTHHalf))

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_medianall"
  p_medianall :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> IO (CTHHalf))

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THHalfTensor_median"
  p_median :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHHalfTensor) -> CInt -> CInt -> IO (()))