{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMathPointwise where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_sigmoid :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_sigmoid"
  c_sigmoid :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_log :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_log"
  c_log :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_lgamma :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_lgamma"
  c_lgamma :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_digamma :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_digamma"
  c_digamma :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_polygamma :  state self n src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_polygamma"
  c_polygamma :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_log1p :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_log1p"
  c_log1p :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_exp :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_exp"
  c_exp :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_expm1 :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_expm1"
  c_expm1 :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cos :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cos"
  c_cos :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_acos :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_acos"
  c_acos :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cosh :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cosh"
  c_cosh :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_sin :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_sin"
  c_sin :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_asin :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_asin"
  c_asin :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_sinh :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_sinh"
  c_sinh :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_tan :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_tan"
  c_tan :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_atan :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_atan"
  c_atan :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_atan2 :  state r_ tx ty -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_atan2"
  c_atan2 :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_tanh :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_tanh"
  c_tanh :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_erf :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_erf"
  c_erf :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_erfinv :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_erfinv"
  c_erfinv :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_pow :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_pow"
  c_pow :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_tpow :  state self value src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_tpow"
  c_tpow :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_sqrt :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_sqrt"
  c_sqrt :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_rsqrt :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_rsqrt"
  c_rsqrt :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_ceil :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_ceil"
  c_ceil :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_floor :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_floor"
  c_floor :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_round :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_round"
  c_round :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_trunc :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_trunc"
  c_trunc :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_frac :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_frac"
  c_frac :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_lerp :  state result a b w -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_lerp"
  c_lerp :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_cinv :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cinv"
  c_cinv :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_neg :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_neg"
  c_neg :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_abs :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_abs"
  c_abs :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_sign :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_sign"
  c_sign :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_clamp :  state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_clamp"
  c_clamp :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_cross :  state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cross"
  c_cross :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_cadd :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cadd"
  c_cadd :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_csub :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_csub"
  c_csub :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cmul :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cmul"
  c_cmul :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cpow :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cpow"
  c_cpow :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cdiv :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cdiv"
  c_cdiv :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_clshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_clshift"
  c_clshift :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_crshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_crshift"
  c_crshift :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cmax :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cmax"
  c_cmax :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cmin :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cmin"
  c_cmin :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cfmod :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cfmod"
  c_cfmod :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cremainder :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cremainder"
  c_cremainder :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cmaxValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cmaxValue"
  c_cmaxValue :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_cminValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cminValue"
  c_cminValue :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_cbitand :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cbitand"
  c_cbitand :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cbitor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cbitor"
  c_cbitor :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_cbitxor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_cbitxor"
  c_cbitxor :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_addcmul :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_addcmul"
  c_addcmul :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_addcdiv :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCudaDoubleTensor_addcdiv"
  c_addcdiv :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | p_sigmoid : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_sigmoid"
  p_sigmoid :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_log : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_log"
  p_log :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_lgamma : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_lgamma"
  p_lgamma :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_digamma : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_digamma"
  p_digamma :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_polygamma : Pointer to function : state self n src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_polygamma"
  p_polygamma :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_log1p : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_log1p"
  p_log1p :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_exp : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_exp"
  p_exp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_expm1 : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_expm1"
  p_expm1 :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cos : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cos"
  p_cos :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_acos : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_acos"
  p_acos :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cosh : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cosh"
  p_cosh :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_sin : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_sin"
  p_sin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_asin : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_asin"
  p_asin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_sinh : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_sinh"
  p_sinh :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_tan : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_tan"
  p_tan :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_atan : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_atan"
  p_atan :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_atan2 : Pointer to function : state r_ tx ty -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_atan2"
  p_atan2 :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_tanh : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_tanh"
  p_tanh :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_erf : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_erf"
  p_erf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_erfinv : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_erfinv"
  p_erfinv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_pow : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_pow"
  p_pow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_tpow : Pointer to function : state self value src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_tpow"
  p_tpow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_sqrt : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_sqrt"
  p_sqrt :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_rsqrt : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_rsqrt"
  p_rsqrt :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_ceil : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_ceil"
  p_ceil :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_floor : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_floor"
  p_floor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_round : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_round"
  p_round :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_trunc : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_trunc"
  p_trunc :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_frac : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_frac"
  p_frac :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_lerp : Pointer to function : state result a b w -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_lerp"
  p_lerp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_cinv : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cinv"
  p_cinv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_neg : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_neg"
  p_neg :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_abs : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_abs"
  p_abs :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_sign : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_sign"
  p_sign :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_clamp : Pointer to function : state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_clamp"
  p_clamp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_cross : Pointer to function : state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cross"
  p_cross :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_cadd : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cadd"
  p_cadd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_csub : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_csub"
  p_csub :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cmul : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cmul"
  p_cmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cpow : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cpow"
  p_cpow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cdiv : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cdiv"
  p_cdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_clshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_clshift"
  p_clshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_crshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_crshift"
  p_crshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cmax : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cmax"
  p_cmax :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cmin : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cmin"
  p_cmin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cfmod : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cfmod"
  p_cfmod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cremainder : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cremainder"
  p_cremainder :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cmaxValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_cminValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cminValue"
  p_cminValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_cbitand : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cbitand"
  p_cbitand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cbitor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cbitor"
  p_cbitor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_cbitxor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_addcmul : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_addcmul"
  p_addcmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_addcdiv : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCudaDoubleTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CDouble -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())