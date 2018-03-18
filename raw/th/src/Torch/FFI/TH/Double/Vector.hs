{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.Vector where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THDoubleVector_fill"
  c_fill_ :: Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill = const c_fill_

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THDoubleVector_cadd"
  c_cadd_ :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd = const c_cadd_

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THDoubleVector_adds"
  c_adds_ :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- | alias of c_adds_ with unused argument (for CTHState) to unify backpack signatures.
c_adds = const c_adds_

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THDoubleVector_cmul"
  c_cmul_ :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul = const c_cmul_

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THDoubleVector_muls"
  c_muls_ :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- | alias of c_muls_ with unused argument (for CTHState) to unify backpack signatures.
c_muls = const c_muls_

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THDoubleVector_cdiv"
  c_cdiv_ :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv = const c_cdiv_

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THDoubleVector_divs"
  c_divs_ :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- | alias of c_divs_ with unused argument (for CTHState) to unify backpack signatures.
c_divs = const c_divs_

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_copy"
  c_copy_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy = const c_copy_

-- | c_neg :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_neg"
  c_neg_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_neg_ with unused argument (for CTHState) to unify backpack signatures.
c_neg = const c_neg_

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THDoubleVector_normal_fill"
  c_normal_fill_ :: Ptr CDouble -> CLLong -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | alias of c_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_fill = const c_normal_fill_

-- | c_abs :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_abs"
  c_abs_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_abs_ with unused argument (for CTHState) to unify backpack signatures.
c_abs = const c_abs_

-- | c_log :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_log"
  c_log_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_log_ with unused argument (for CTHState) to unify backpack signatures.
c_log = const c_log_

-- | c_lgamma :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_lgamma"
  c_lgamma_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_lgamma_ with unused argument (for CTHState) to unify backpack signatures.
c_lgamma = const c_lgamma_

-- | c_digamma :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_digamma"
  c_digamma_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_digamma_ with unused argument (for CTHState) to unify backpack signatures.
c_digamma = const c_digamma_

-- | c_trigamma :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_trigamma"
  c_trigamma_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_trigamma_ with unused argument (for CTHState) to unify backpack signatures.
c_trigamma = const c_trigamma_

-- | c_log1p :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_log1p"
  c_log1p_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_log1p_ with unused argument (for CTHState) to unify backpack signatures.
c_log1p = const c_log1p_

-- | c_sigmoid :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_sigmoid"
  c_sigmoid_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_sigmoid_ with unused argument (for CTHState) to unify backpack signatures.
c_sigmoid = const c_sigmoid_

-- | c_exp :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_exp"
  c_exp_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_exp_ with unused argument (for CTHState) to unify backpack signatures.
c_exp = const c_exp_

-- | c_expm1 :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_expm1"
  c_expm1_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_expm1_ with unused argument (for CTHState) to unify backpack signatures.
c_expm1 = const c_expm1_

-- | c_erf :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_erf"
  c_erf_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_erf_ with unused argument (for CTHState) to unify backpack signatures.
c_erf = const c_erf_

-- | c_erfinv :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_erfinv"
  c_erfinv_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_erfinv_ with unused argument (for CTHState) to unify backpack signatures.
c_erfinv = const c_erfinv_

-- | c_cos :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_cos"
  c_cos_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_cos_ with unused argument (for CTHState) to unify backpack signatures.
c_cos = const c_cos_

-- | c_acos :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_acos"
  c_acos_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_acos_ with unused argument (for CTHState) to unify backpack signatures.
c_acos = const c_acos_

-- | c_cosh :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_cosh"
  c_cosh_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_cosh_ with unused argument (for CTHState) to unify backpack signatures.
c_cosh = const c_cosh_

-- | c_sin :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_sin"
  c_sin_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_sin_ with unused argument (for CTHState) to unify backpack signatures.
c_sin = const c_sin_

-- | c_asin :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_asin"
  c_asin_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_asin_ with unused argument (for CTHState) to unify backpack signatures.
c_asin = const c_asin_

-- | c_sinh :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_sinh"
  c_sinh_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_sinh_ with unused argument (for CTHState) to unify backpack signatures.
c_sinh = const c_sinh_

-- | c_tan :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_tan"
  c_tan_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_tan_ with unused argument (for CTHState) to unify backpack signatures.
c_tan = const c_tan_

-- | c_atan :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_atan"
  c_atan_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_atan_ with unused argument (for CTHState) to unify backpack signatures.
c_atan = const c_atan_

-- | c_tanh :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_tanh"
  c_tanh_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_tanh_ with unused argument (for CTHState) to unify backpack signatures.
c_tanh = const c_tanh_

-- | c_pow :  y x c n -> void
foreign import ccall "THVector.h THDoubleVector_pow"
  c_pow_ :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- | alias of c_pow_ with unused argument (for CTHState) to unify backpack signatures.
c_pow = const c_pow_

-- | c_sqrt :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_sqrt"
  c_sqrt_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_sqrt_ with unused argument (for CTHState) to unify backpack signatures.
c_sqrt = const c_sqrt_

-- | c_rsqrt :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_rsqrt"
  c_rsqrt_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_rsqrt_ with unused argument (for CTHState) to unify backpack signatures.
c_rsqrt = const c_rsqrt_

-- | c_ceil :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_ceil"
  c_ceil_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_ceil_ with unused argument (for CTHState) to unify backpack signatures.
c_ceil = const c_ceil_

-- | c_floor :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_floor"
  c_floor_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_floor_ with unused argument (for CTHState) to unify backpack signatures.
c_floor = const c_floor_

-- | c_round :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_round"
  c_round_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_round_ with unused argument (for CTHState) to unify backpack signatures.
c_round = const c_round_

-- | c_trunc :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_trunc"
  c_trunc_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_trunc_ with unused argument (for CTHState) to unify backpack signatures.
c_trunc = const c_trunc_

-- | c_frac :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_frac"
  c_frac_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_frac_ with unused argument (for CTHState) to unify backpack signatures.
c_frac = const c_frac_

-- | c_cinv :  y x n -> void
foreign import ccall "THVector.h THDoubleVector_cinv"
  c_cinv_ :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- | alias of c_cinv_ with unused argument (for CTHState) to unify backpack signatures.
c_cinv = const c_cinv_

-- | c_vectorDispatchInit :   -> void
foreign import ccall "THVector.h THDoubleVector_vectorDispatchInit"
  c_vectorDispatchInit_ :: IO ()

-- | alias of c_vectorDispatchInit_ with unused argument (for CTHState) to unify backpack signatures.
c_vectorDispatchInit = const c_vectorDispatchInit_

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THDoubleVector_fill"
  p_fill_ :: FunPtr (Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- | alias of p_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_fill = const p_fill_

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THDoubleVector_cadd"
  p_cadd_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- | alias of p_cadd_ with unused argument (for CTHState) to unify backpack signatures.
p_cadd = const p_cadd_

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THDoubleVector_adds"
  p_adds_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- | alias of p_adds_ with unused argument (for CTHState) to unify backpack signatures.
p_adds = const p_adds_

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THDoubleVector_cmul"
  p_cmul_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_cmul_ with unused argument (for CTHState) to unify backpack signatures.
p_cmul = const p_cmul_

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THDoubleVector_muls"
  p_muls_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- | alias of p_muls_ with unused argument (for CTHState) to unify backpack signatures.
p_muls = const p_muls_

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THDoubleVector_cdiv"
  p_cdiv_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
p_cdiv = const p_cdiv_

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THDoubleVector_divs"
  p_divs_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- | alias of p_divs_ with unused argument (for CTHState) to unify backpack signatures.
p_divs = const p_divs_

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_copy"
  p_copy_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_copy_ with unused argument (for CTHState) to unify backpack signatures.
p_copy = const p_copy_

-- | p_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_neg"
  p_neg_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_neg_ with unused argument (for CTHState) to unify backpack signatures.
p_neg = const p_neg_

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THDoubleVector_normal_fill"
  p_normal_fill_ :: FunPtr (Ptr CDouble -> CLLong -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | alias of p_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_normal_fill = const p_normal_fill_

-- | p_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_abs"
  p_abs_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_abs_ with unused argument (for CTHState) to unify backpack signatures.
p_abs = const p_abs_

-- | p_log : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_log"
  p_log_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_log_ with unused argument (for CTHState) to unify backpack signatures.
p_log = const p_log_

-- | p_lgamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_lgamma"
  p_lgamma_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_lgamma_ with unused argument (for CTHState) to unify backpack signatures.
p_lgamma = const p_lgamma_

-- | p_digamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_digamma"
  p_digamma_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_digamma_ with unused argument (for CTHState) to unify backpack signatures.
p_digamma = const p_digamma_

-- | p_trigamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_trigamma"
  p_trigamma_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_trigamma_ with unused argument (for CTHState) to unify backpack signatures.
p_trigamma = const p_trigamma_

-- | p_log1p : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_log1p"
  p_log1p_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_log1p_ with unused argument (for CTHState) to unify backpack signatures.
p_log1p = const p_log1p_

-- | p_sigmoid : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_sigmoid"
  p_sigmoid_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_sigmoid_ with unused argument (for CTHState) to unify backpack signatures.
p_sigmoid = const p_sigmoid_

-- | p_exp : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_exp"
  p_exp_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_exp_ with unused argument (for CTHState) to unify backpack signatures.
p_exp = const p_exp_

-- | p_expm1 : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_expm1"
  p_expm1_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_expm1_ with unused argument (for CTHState) to unify backpack signatures.
p_expm1 = const p_expm1_

-- | p_erf : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_erf"
  p_erf_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_erf_ with unused argument (for CTHState) to unify backpack signatures.
p_erf = const p_erf_

-- | p_erfinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_erfinv"
  p_erfinv_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_erfinv_ with unused argument (for CTHState) to unify backpack signatures.
p_erfinv = const p_erfinv_

-- | p_cos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_cos"
  p_cos_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_cos_ with unused argument (for CTHState) to unify backpack signatures.
p_cos = const p_cos_

-- | p_acos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_acos"
  p_acos_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_acos_ with unused argument (for CTHState) to unify backpack signatures.
p_acos = const p_acos_

-- | p_cosh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_cosh"
  p_cosh_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_cosh_ with unused argument (for CTHState) to unify backpack signatures.
p_cosh = const p_cosh_

-- | p_sin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_sin"
  p_sin_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_sin_ with unused argument (for CTHState) to unify backpack signatures.
p_sin = const p_sin_

-- | p_asin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_asin"
  p_asin_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_asin_ with unused argument (for CTHState) to unify backpack signatures.
p_asin = const p_asin_

-- | p_sinh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_sinh"
  p_sinh_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_sinh_ with unused argument (for CTHState) to unify backpack signatures.
p_sinh = const p_sinh_

-- | p_tan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_tan"
  p_tan_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_tan_ with unused argument (for CTHState) to unify backpack signatures.
p_tan = const p_tan_

-- | p_atan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_atan"
  p_atan_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_atan_ with unused argument (for CTHState) to unify backpack signatures.
p_atan = const p_atan_

-- | p_tanh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_tanh"
  p_tanh_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_tanh_ with unused argument (for CTHState) to unify backpack signatures.
p_tanh = const p_tanh_

-- | p_pow : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THDoubleVector_pow"
  p_pow_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- | alias of p_pow_ with unused argument (for CTHState) to unify backpack signatures.
p_pow = const p_pow_

-- | p_sqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_sqrt"
  p_sqrt_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_sqrt_ with unused argument (for CTHState) to unify backpack signatures.
p_sqrt = const p_sqrt_

-- | p_rsqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_rsqrt"
  p_rsqrt_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_rsqrt_ with unused argument (for CTHState) to unify backpack signatures.
p_rsqrt = const p_rsqrt_

-- | p_ceil : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_ceil"
  p_ceil_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_ceil_ with unused argument (for CTHState) to unify backpack signatures.
p_ceil = const p_ceil_

-- | p_floor : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_floor"
  p_floor_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_floor_ with unused argument (for CTHState) to unify backpack signatures.
p_floor = const p_floor_

-- | p_round : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_round"
  p_round_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_round_ with unused argument (for CTHState) to unify backpack signatures.
p_round = const p_round_

-- | p_trunc : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_trunc"
  p_trunc_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_trunc_ with unused argument (for CTHState) to unify backpack signatures.
p_trunc = const p_trunc_

-- | p_frac : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_frac"
  p_frac_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_frac_ with unused argument (for CTHState) to unify backpack signatures.
p_frac = const p_frac_

-- | p_cinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_cinv"
  p_cinv_ :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- | alias of p_cinv_ with unused argument (for CTHState) to unify backpack signatures.
p_cinv = const p_cinv_

-- | p_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THDoubleVector_vectorDispatchInit"
  p_vectorDispatchInit_ :: FunPtr (IO ())

-- | alias of p_vectorDispatchInit_ with unused argument (for CTHState) to unify backpack signatures.
p_vectorDispatchInit = const p_vectorDispatchInit_