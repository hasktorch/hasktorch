{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.TensorMathReduce
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
foreign import ccall "THCTensorMathReduce.h THShortTensor_sum"
  c_sum :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (())

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THShortTensor_prod"
  c_prod :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (())

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THShortTensor_sumall"
  c_sumall :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CLong)

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THShortTensor_prodall"
  c_prodall :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CLong)

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THShortTensor_min"
  c_min :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (())

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THShortTensor_max"
  c_max :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (())

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THShortTensor_minall"
  c_minall :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CShort)

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THShortTensor_maxall"
  c_maxall :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CShort)

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THShortTensor_medianall"
  c_medianall :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CShort)

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THShortTensor_median"
  c_median :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (())

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THShortTensor_sum"
  p_sum :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (()))

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THShortTensor_prod"
  p_prod :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (()))

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THShortTensor_sumall"
  p_sumall :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CLong))

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THShortTensor_prodall"
  p_prodall :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CLong))

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THShortTensor_min"
  p_min :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (()))

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THShortTensor_max"
  p_max :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (()))

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THShortTensor_minall"
  p_minall :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CShort))

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THShortTensor_maxall"
  p_maxall :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CShort))

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THShortTensor_medianall"
  p_medianall :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> IO (CShort))

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THShortTensor_median"
  p_median :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHShortTensor) -> CInt -> CInt -> IO (()))