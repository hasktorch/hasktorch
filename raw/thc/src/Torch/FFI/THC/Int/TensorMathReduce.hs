{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorMathReduce
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
foreign import ccall "THCTensorMathReduce.h THCIntTensor_sum"
  c_sum :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ()

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCIntTensor_prod"
  c_prod :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ()

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCIntTensor_sumall"
  c_sumall :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CLong

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCIntTensor_prodall"
  c_prodall :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CLong

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCIntTensor_min"
  c_min :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ()

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCIntTensor_max"
  c_max :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ()

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCIntTensor_minall"
  c_minall :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CInt

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCIntTensor_maxall"
  c_maxall :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CInt

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCIntTensor_medianall"
  c_medianall :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CInt

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCIntTensor_median"
  c_median :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ()

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_sum"
  p_sum :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_prod"
  p_prod :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ())

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_sumall"
  p_sumall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CLong)

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_prodall"
  p_prodall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CLong)

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_min"
  p_min :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_max"
  p_max :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ())

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_minall"
  p_minall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CInt)

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_maxall"
  p_maxall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CInt)

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_medianall"
  p_medianall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> IO CInt)

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCIntTensor_median"
  p_median :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> CInt -> CInt -> IO ())