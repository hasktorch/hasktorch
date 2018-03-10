{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMathReduce
  ( c_renorm
  , c_std
  , c_norm
  , c_var
  , c_stdall
  , c_normall
  , c_varall
  , c_sum
  , c_prod
  , c_mean
  , c_sumall
  , c_prodall
  , c_meanall
  , c_min
  , c_max
  , c_minall
  , c_maxall
  , c_medianall
  , c_median
  , c_dist
  , p_renorm
  , p_std
  , p_norm
  , p_var
  , p_stdall
  , p_normall
  , p_varall
  , p_sum
  , p_prod
  , p_mean
  , p_sumall
  , p_prodall
  , p_meanall
  , p_min
  , p_max
  , p_minall
  , p_maxall
  , p_medianall
  , p_median
  , p_dist
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_renorm :  state self src value dimension max_norm -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_renorm"
  c_renorm :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> CInt -> CFloat -> IO ()

-- | c_std :  state self src dim biased keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_std"
  c_std :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_norm :  state self src value dimension keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_norm"
  c_norm :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> CInt -> CInt -> IO ()

-- | c_var :  state self src dim biased keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_var"
  c_var :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> CInt -> IO ()

-- | c_stdall :  state self biased -> accreal
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_stdall"
  c_stdall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> IO CDouble

-- | c_normall :  state self value -> accreal
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_normall"
  c_normall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> IO CDouble

-- | c_varall :  state self biased -> accreal
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_varall"
  c_varall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> IO CDouble

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_sum"
  c_sum :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_prod"
  c_prod :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_mean :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_mean"
  c_mean :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_sumall"
  c_sumall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CDouble

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_prodall"
  c_prodall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CDouble

-- | c_meanall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_meanall"
  c_meanall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CDouble

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_min"
  c_min :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_max"
  c_max :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_minall"
  c_minall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CFloat

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_maxall"
  c_maxall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CFloat

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_medianall"
  c_medianall :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CFloat

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_median"
  c_median :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_dist :  state self src value -> accreal
foreign import ccall "THCTensorMathReduce.h THCFloatTensor_dist"
  c_dist :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO CDouble

-- | p_renorm : Pointer to function : state self src value dimension max_norm -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_renorm"
  p_renorm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> CInt -> CFloat -> IO ())

-- | p_std : Pointer to function : state self src dim biased keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_std"
  p_std :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_norm : Pointer to function : state self src value dimension keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_norm"
  p_norm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> CInt -> CInt -> IO ())

-- | p_var : Pointer to function : state self src dim biased keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_var"
  p_var :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> CInt -> IO ())

-- | p_stdall : Pointer to function : state self biased -> accreal
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_stdall"
  p_stdall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> IO CDouble)

-- | p_normall : Pointer to function : state self value -> accreal
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_normall"
  p_normall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> IO CDouble)

-- | p_varall : Pointer to function : state self biased -> accreal
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_varall"
  p_varall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CInt -> IO CDouble)

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_sum"
  p_sum :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_prod"
  p_prod :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_mean : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_mean"
  p_mean :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_sumall"
  p_sumall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CDouble)

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_prodall"
  p_prodall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CDouble)

-- | p_meanall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_meanall"
  p_meanall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CDouble)

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_min"
  p_min :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_max"
  p_max :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_minall"
  p_minall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CFloat)

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_maxall"
  p_maxall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CFloat)

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_medianall"
  p_medianall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO CFloat)

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_median"
  p_median :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_dist : Pointer to function : state self src value -> accreal
foreign import ccall "THCTensorMathReduce.h &THCFloatTensor_dist"
  p_dist :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CFloat -> IO CDouble)