{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMathPointwise where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_pow :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_pow"
  c_pow :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> IO ()

-- | c_tpow :  state self value src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_tpow"
  c_tpow :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cpow :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cpow"
  c_cpow :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_sigmoid :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_sigmoid"
  c_sigmoid :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_log :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_log"
  c_log :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_lgamma :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_lgamma"
  c_lgamma :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_digamma :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_digamma"
  c_digamma :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_polygamma :  state self n src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_polygamma"
  c_polygamma :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CLLong -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_log1p :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_log1p"
  c_log1p :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_exp :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_exp"
  c_exp :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_expm1 :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_expm1"
  c_expm1 :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cos :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cos"
  c_cos :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_acos :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_acos"
  c_acos :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cosh :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cosh"
  c_cosh :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_sin :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_sin"
  c_sin :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_asin :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_asin"
  c_asin :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_sinh :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_sinh"
  c_sinh :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_tan :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_tan"
  c_tan :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_atan :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_atan"
  c_atan :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_atan2 :  state r_ tx ty -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_atan2"
  c_atan2 :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_tanh :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_tanh"
  c_tanh :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_erf :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_erf"
  c_erf :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_erfinv :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_erfinv"
  c_erfinv :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_sqrt :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_sqrt"
  c_sqrt :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_rsqrt :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_rsqrt"
  c_rsqrt :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_ceil :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_ceil"
  c_ceil :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_floor :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_floor"
  c_floor :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_round :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_round"
  c_round :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_trunc :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_trunc"
  c_trunc :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_frac :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_frac"
  c_frac :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_lerp :  state result a b w -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_lerp"
  c_lerp :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> IO ()

-- | c_cinv :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cinv"
  c_cinv :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_neg :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_neg"
  c_neg :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_abs :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_abs"
  c_abs :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_sign :  state self src -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_sign"
  c_sign :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_clamp :  state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_clamp"
  c_clamp :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> CFloat -> IO ()

-- | c_cross :  state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cross"
  c_cross :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_cadd :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cadd"
  c_cadd :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_csub :  state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_csub"
  c_csub :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cmul :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cmul"
  c_cmul :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cdiv :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cdiv"
  c_cdiv :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_clshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_clshift"
  c_clshift :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_crshift :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_crshift"
  c_crshift :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cmax :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cmax"
  c_cmax :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cmin :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cmin"
  c_cmin :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cfmod :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cfmod"
  c_cfmod :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cremainder :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cremainder"
  c_cremainder :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cmaxValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cmaxValue"
  c_cmaxValue :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> IO ()

-- | c_cminValue :  state self src value -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cminValue"
  c_cminValue :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> IO ()

-- | c_cbitand :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cbitand"
  c_cbitand :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cbitor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cbitor"
  c_cbitor :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_cbitxor :  state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_cbitxor"
  c_cbitxor :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_addcmul :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_addcmul"
  c_addcmul :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_addcdiv :  state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h THCFloatTensor_addcdiv"
  c_addcdiv :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | p_pow : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_pow"
  p_pow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> IO ())

-- | p_tpow : Pointer to function : state self value src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_tpow"
  p_tpow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cpow : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cpow"
  p_cpow :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_sigmoid : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_sigmoid"
  p_sigmoid :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_log : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_log"
  p_log :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_lgamma : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_lgamma"
  p_lgamma :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_digamma : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_digamma"
  p_digamma :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_polygamma : Pointer to function : state self n src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_polygamma"
  p_polygamma :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CLLong -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_log1p : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_log1p"
  p_log1p :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_exp : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_exp"
  p_exp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_expm1 : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_expm1"
  p_expm1 :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cos : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cos"
  p_cos :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_acos : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_acos"
  p_acos :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cosh : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cosh"
  p_cosh :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_sin : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_sin"
  p_sin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_asin : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_asin"
  p_asin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_sinh : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_sinh"
  p_sinh :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_tan : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_tan"
  p_tan :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_atan : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_atan"
  p_atan :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_atan2 : Pointer to function : state r_ tx ty -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_atan2"
  p_atan2 :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_tanh : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_tanh"
  p_tanh :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_erf : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_erf"
  p_erf :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_erfinv : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_erfinv"
  p_erfinv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_sqrt : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_sqrt"
  p_sqrt :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_rsqrt : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_rsqrt"
  p_rsqrt :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_ceil : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_ceil"
  p_ceil :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_floor : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_floor"
  p_floor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_round : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_round"
  p_round :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_trunc : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_trunc"
  p_trunc :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_frac : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_frac"
  p_frac :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_lerp : Pointer to function : state result a b w -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_lerp"
  p_lerp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> IO ())

-- | p_cinv : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cinv"
  p_cinv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_neg : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_neg"
  p_neg :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_abs : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_abs"
  p_abs :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_sign : Pointer to function : state self src -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_sign"
  p_sign :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_clamp : Pointer to function : state self src min_value max_value -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_clamp"
  p_clamp :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> CFloat -> IO ())

-- | p_cross : Pointer to function : state self src1 src2 dimension -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cross"
  p_cross :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_cadd : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cadd"
  p_cadd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_csub : Pointer to function : state self src1 value src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_csub"
  p_csub :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cmul : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cmul"
  p_cmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cdiv : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cdiv"
  p_cdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_clshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_clshift"
  p_clshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_crshift : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_crshift"
  p_crshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cmax : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cmax"
  p_cmax :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cmin : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cmin"
  p_cmin :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cfmod : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cfmod"
  p_cfmod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cremainder : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cremainder"
  p_cremainder :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cmaxValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cmaxValue"
  p_cmaxValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> IO ())

-- | p_cminValue : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cminValue"
  p_cminValue :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> IO ())

-- | p_cbitand : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cbitand"
  p_cbitand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cbitor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cbitor"
  p_cbitor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_cbitxor : Pointer to function : state self src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_cbitxor"
  p_cbitxor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_addcmul : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_addcmul"
  p_addcmul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_addcdiv : Pointer to function : state self t value src1 src2 -> void
foreign import ccall "THCTensorMathPointwise.h &THCFloatTensor_addcdiv"
  p_addcdiv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CFloat -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())