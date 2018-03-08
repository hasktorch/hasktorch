{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorMathReduce
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
foreign import ccall "THCTensorMathReduce.h THFloatTensor_renorm"
  c_renorm :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CInt -> CFloat -> IO (())

-- | c_std :  state self src dim biased keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_std"
  c_std :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> CInt -> IO (())

-- | c_norm :  state self src value dimension keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_norm"
  c_norm :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CInt -> CInt -> IO (())

-- | c_var :  state self src dim biased keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_var"
  c_var :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> CInt -> IO (())

-- | c_stdall :  state self biased -> accreal
foreign import ccall "THCTensorMathReduce.h THFloatTensor_stdall"
  c_stdall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CInt -> IO (CDouble)

-- | c_normall :  state self value -> accreal
foreign import ccall "THCTensorMathReduce.h THFloatTensor_normall"
  c_normall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> IO (CDouble)

-- | c_varall :  state self biased -> accreal
foreign import ccall "THCTensorMathReduce.h THFloatTensor_varall"
  c_varall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CInt -> IO (CDouble)

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_sum"
  c_sum :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_prod"
  c_prod :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_mean :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_mean"
  c_mean :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THFloatTensor_sumall"
  c_sumall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THFloatTensor_prodall"
  c_prodall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_meanall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THFloatTensor_meanall"
  c_meanall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CDouble)

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_min"
  c_min :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_max"
  c_max :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THFloatTensor_minall"
  c_minall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CFloat)

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THFloatTensor_maxall"
  c_maxall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CFloat)

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THFloatTensor_medianall"
  c_medianall :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CFloat)

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THFloatTensor_median"
  c_median :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (())

-- | c_dist :  state self src value -> accreal
foreign import ccall "THCTensorMathReduce.h THFloatTensor_dist"
  c_dist :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (CDouble)

-- | p_renorm : Pointer to function : state self src value dimension max_norm -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_renorm"
  p_renorm :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CInt -> CFloat -> IO (()))

-- | p_std : Pointer to function : state self src dim biased keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_std"
  p_std :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> CInt -> IO (()))

-- | p_norm : Pointer to function : state self src value dimension keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_norm"
  p_norm :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> CInt -> CInt -> IO (()))

-- | p_var : Pointer to function : state self src dim biased keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_var"
  p_var :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> CInt -> IO (()))

-- | p_stdall : Pointer to function : state self biased -> accreal
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_stdall"
  p_stdall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CInt -> IO (CDouble))

-- | p_normall : Pointer to function : state self value -> accreal
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_normall"
  p_normall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CFloat -> IO (CDouble))

-- | p_varall : Pointer to function : state self biased -> accreal
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_varall"
  p_varall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> CInt -> IO (CDouble))

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_sum"
  p_sum :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_prod"
  p_prod :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_mean : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_mean"
  p_mean :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_sumall"
  p_sumall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_prodall"
  p_prodall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_meanall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_meanall"
  p_meanall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CDouble))

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_min"
  p_min :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_max"
  p_max :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_minall"
  p_minall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CFloat))

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_maxall"
  p_maxall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CFloat))

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_medianall"
  p_medianall :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> IO (CFloat))

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_median"
  p_median :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CInt -> CInt -> IO (()))

-- | p_dist : Pointer to function : state self src value -> accreal
foreign import ccall "THCTensorMathReduce.h &THFloatTensor_dist"
  p_dist :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CFloat -> IO (CDouble))