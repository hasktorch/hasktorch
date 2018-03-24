{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.Vector where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THFloatVector_fill"
  c_fill_ :: Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill :: Ptr C'THState -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()
c_fill = const c_fill_

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THFloatVector_cadd"
  c_cadd_ :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()
c_cadd = const c_cadd_

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THFloatVector_adds"
  c_adds_ :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- | alias of c_adds_ with unused argument (for CTHState) to unify backpack signatures.
c_adds :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()
c_adds = const c_adds_

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THFloatVector_cmul"
  c_cmul_ :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_cmul = const c_cmul_

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THFloatVector_muls"
  c_muls_ :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- | alias of c_muls_ with unused argument (for CTHState) to unify backpack signatures.
c_muls :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()
c_muls = const c_muls_

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THFloatVector_cdiv"
  c_cdiv_ :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_cdiv = const c_cdiv_

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THFloatVector_divs"
  c_divs_ :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- | alias of c_divs_ with unused argument (for CTHState) to unify backpack signatures.
c_divs :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()
c_divs = const c_divs_

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THFloatVector_copy"
  c_copy_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_copy = const c_copy_

-- | c_neg :  y x n -> void
foreign import ccall "THVector.h THFloatVector_neg"
  c_neg_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_neg_ with unused argument (for CTHState) to unify backpack signatures.
c_neg :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_neg = const c_neg_

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THFloatVector_normal_fill"
  c_normal_fill_ :: Ptr CFloat -> CLLong -> Ptr C'THGenerator -> CFloat -> CFloat -> IO ()

-- | alias of c_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_fill :: Ptr C'THState -> Ptr CFloat -> CLLong -> Ptr C'THGenerator -> CFloat -> CFloat -> IO ()
c_normal_fill = const c_normal_fill_

-- | c_abs :  y x n -> void
foreign import ccall "THVector.h THFloatVector_abs"
  c_abs_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_abs_ with unused argument (for CTHState) to unify backpack signatures.
c_abs :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_abs = const c_abs_

-- | c_log :  y x n -> void
foreign import ccall "THVector.h THFloatVector_log"
  c_log_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_log_ with unused argument (for CTHState) to unify backpack signatures.
c_log :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_log = const c_log_

-- | c_lgamma :  y x n -> void
foreign import ccall "THVector.h THFloatVector_lgamma"
  c_lgamma_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_lgamma_ with unused argument (for CTHState) to unify backpack signatures.
c_lgamma :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_lgamma = const c_lgamma_

-- | c_digamma :  y x n -> void
foreign import ccall "THVector.h THFloatVector_digamma"
  c_digamma_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_digamma_ with unused argument (for CTHState) to unify backpack signatures.
c_digamma :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_digamma = const c_digamma_

-- | c_trigamma :  y x n -> void
foreign import ccall "THVector.h THFloatVector_trigamma"
  c_trigamma_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_trigamma_ with unused argument (for CTHState) to unify backpack signatures.
c_trigamma :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_trigamma = const c_trigamma_

-- | c_log1p :  y x n -> void
foreign import ccall "THVector.h THFloatVector_log1p"
  c_log1p_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_log1p_ with unused argument (for CTHState) to unify backpack signatures.
c_log1p :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_log1p = const c_log1p_

-- | c_sigmoid :  y x n -> void
foreign import ccall "THVector.h THFloatVector_sigmoid"
  c_sigmoid_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_sigmoid_ with unused argument (for CTHState) to unify backpack signatures.
c_sigmoid :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_sigmoid = const c_sigmoid_

-- | c_exp :  y x n -> void
foreign import ccall "THVector.h THFloatVector_exp"
  c_exp_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_exp_ with unused argument (for CTHState) to unify backpack signatures.
c_exp :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_exp = const c_exp_

-- | c_expm1 :  y x n -> void
foreign import ccall "THVector.h THFloatVector_expm1"
  c_expm1_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_expm1_ with unused argument (for CTHState) to unify backpack signatures.
c_expm1 :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_expm1 = const c_expm1_

-- | c_erf :  y x n -> void
foreign import ccall "THVector.h THFloatVector_erf"
  c_erf_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_erf_ with unused argument (for CTHState) to unify backpack signatures.
c_erf :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_erf = const c_erf_

-- | c_erfinv :  y x n -> void
foreign import ccall "THVector.h THFloatVector_erfinv"
  c_erfinv_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_erfinv_ with unused argument (for CTHState) to unify backpack signatures.
c_erfinv :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_erfinv = const c_erfinv_

-- | c_cos :  y x n -> void
foreign import ccall "THVector.h THFloatVector_cos"
  c_cos_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_cos_ with unused argument (for CTHState) to unify backpack signatures.
c_cos :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_cos = const c_cos_

-- | c_acos :  y x n -> void
foreign import ccall "THVector.h THFloatVector_acos"
  c_acos_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_acos_ with unused argument (for CTHState) to unify backpack signatures.
c_acos :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_acos = const c_acos_

-- | c_cosh :  y x n -> void
foreign import ccall "THVector.h THFloatVector_cosh"
  c_cosh_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_cosh_ with unused argument (for CTHState) to unify backpack signatures.
c_cosh :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_cosh = const c_cosh_

-- | c_sin :  y x n -> void
foreign import ccall "THVector.h THFloatVector_sin"
  c_sin_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_sin_ with unused argument (for CTHState) to unify backpack signatures.
c_sin :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_sin = const c_sin_

-- | c_asin :  y x n -> void
foreign import ccall "THVector.h THFloatVector_asin"
  c_asin_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_asin_ with unused argument (for CTHState) to unify backpack signatures.
c_asin :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_asin = const c_asin_

-- | c_sinh :  y x n -> void
foreign import ccall "THVector.h THFloatVector_sinh"
  c_sinh_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_sinh_ with unused argument (for CTHState) to unify backpack signatures.
c_sinh :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_sinh = const c_sinh_

-- | c_tan :  y x n -> void
foreign import ccall "THVector.h THFloatVector_tan"
  c_tan_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_tan_ with unused argument (for CTHState) to unify backpack signatures.
c_tan :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_tan = const c_tan_

-- | c_atan :  y x n -> void
foreign import ccall "THVector.h THFloatVector_atan"
  c_atan_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_atan_ with unused argument (for CTHState) to unify backpack signatures.
c_atan :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_atan = const c_atan_

-- | c_tanh :  y x n -> void
foreign import ccall "THVector.h THFloatVector_tanh"
  c_tanh_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_tanh_ with unused argument (for CTHState) to unify backpack signatures.
c_tanh :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_tanh = const c_tanh_

-- | c_pow :  y x c n -> void
foreign import ccall "THVector.h THFloatVector_pow"
  c_pow_ :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- | alias of c_pow_ with unused argument (for CTHState) to unify backpack signatures.
c_pow :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()
c_pow = const c_pow_

-- | c_sqrt :  y x n -> void
foreign import ccall "THVector.h THFloatVector_sqrt"
  c_sqrt_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_sqrt_ with unused argument (for CTHState) to unify backpack signatures.
c_sqrt :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_sqrt = const c_sqrt_

-- | c_rsqrt :  y x n -> void
foreign import ccall "THVector.h THFloatVector_rsqrt"
  c_rsqrt_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_rsqrt_ with unused argument (for CTHState) to unify backpack signatures.
c_rsqrt :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_rsqrt = const c_rsqrt_

-- | c_ceil :  y x n -> void
foreign import ccall "THVector.h THFloatVector_ceil"
  c_ceil_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_ceil_ with unused argument (for CTHState) to unify backpack signatures.
c_ceil :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_ceil = const c_ceil_

-- | c_floor :  y x n -> void
foreign import ccall "THVector.h THFloatVector_floor"
  c_floor_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_floor_ with unused argument (for CTHState) to unify backpack signatures.
c_floor :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_floor = const c_floor_

-- | c_round :  y x n -> void
foreign import ccall "THVector.h THFloatVector_round"
  c_round_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_round_ with unused argument (for CTHState) to unify backpack signatures.
c_round :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_round = const c_round_

-- | c_trunc :  y x n -> void
foreign import ccall "THVector.h THFloatVector_trunc"
  c_trunc_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_trunc_ with unused argument (for CTHState) to unify backpack signatures.
c_trunc :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_trunc = const c_trunc_

-- | c_frac :  y x n -> void
foreign import ccall "THVector.h THFloatVector_frac"
  c_frac_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_frac_ with unused argument (for CTHState) to unify backpack signatures.
c_frac :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_frac = const c_frac_

-- | c_cinv :  y x n -> void
foreign import ccall "THVector.h THFloatVector_cinv"
  c_cinv_ :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- | alias of c_cinv_ with unused argument (for CTHState) to unify backpack signatures.
c_cinv :: Ptr C'THState -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()
c_cinv = const c_cinv_

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THFloatVector_fill"
  p_fill :: FunPtr (Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THFloatVector_cadd"
  p_cadd :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THFloatVector_adds"
  p_adds :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THFloatVector_cmul"
  p_cmul :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THFloatVector_muls"
  p_muls :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THFloatVector_cdiv"
  p_cdiv :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THFloatVector_divs"
  p_divs :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_copy"
  p_copy :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_neg"
  p_neg :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THFloatVector_normal_fill"
  p_normal_fill :: FunPtr (Ptr CFloat -> CLLong -> Ptr C'THGenerator -> CFloat -> CFloat -> IO ())

-- | p_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_abs"
  p_abs :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_log : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_log"
  p_log :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_lgamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_lgamma"
  p_lgamma :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_digamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_digamma"
  p_digamma :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_trigamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_trigamma"
  p_trigamma :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_log1p : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_log1p"
  p_log1p :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_sigmoid : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_sigmoid"
  p_sigmoid :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_exp : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_exp"
  p_exp :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_expm1 : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_expm1"
  p_expm1 :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_erf : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_erf"
  p_erf :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_erfinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_erfinv"
  p_erfinv :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_cos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_cos"
  p_cos :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_acos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_acos"
  p_acos :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_cosh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_cosh"
  p_cosh :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_sin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_sin"
  p_sin :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_asin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_asin"
  p_asin :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_sinh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_sinh"
  p_sinh :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_tan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_tan"
  p_tan :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_atan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_atan"
  p_atan :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_tanh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_tanh"
  p_tanh :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_pow : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THFloatVector_pow"
  p_pow :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- | p_sqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_sqrt"
  p_sqrt :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_rsqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_rsqrt"
  p_rsqrt :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_ceil : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_ceil"
  p_ceil :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_floor : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_floor"
  p_floor :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_round : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_round"
  p_round :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_trunc : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_trunc"
  p_trunc :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_frac : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_frac"
  p_frac :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- | p_cinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_cinv"
  p_cinv :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())