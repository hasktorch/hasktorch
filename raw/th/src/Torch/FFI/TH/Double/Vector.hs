{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.Vector
  ( c_fill
  , c_cadd
  , c_adds
  , c_cmul
  , c_muls
  , c_cdiv
  , c_divs
  , c_copy
  , c_neg
  , c_normal_fill
  , c_abs
  , c_log
  , c_lgamma
  , c_digamma
  , c_trigamma
  , c_log1p
  , c_sigmoid
  , c_exp
  , c_expm1
  , c_erf
  , c_erfinv
  , c_cos
  , c_acos
  , c_cosh
  , c_sin
  , c_asin
  , c_sinh
  , c_tan
  , c_atan
  , c_tanh
  , c_pow
  , c_sqrt
  , c_rsqrt
  , c_ceil
  , c_floor
  , c_round
  , c_trunc
  , c_frac
  , c_cinv
  , c_vectorDispatchInit
  , p_fill
  , p_cadd
  , p_adds
  , p_cmul
  , p_muls
  , p_cdiv
  , p_divs
  , p_copy
  , p_neg
  , p_normal_fill
  , p_abs
  , p_log
  , p_lgamma
  , p_digamma
  , p_trigamma
  , p_log1p
  , p_sigmoid
  , p_exp
  , p_expm1
  , p_erf
  , p_erfinv
  , p_cos
  , p_acos
  , p_cosh
  , p_sin
  , p_asin
  , p_sinh
  , p_tan
  , p_atan
  , p_tanh
  , p_pow
  , p_sqrt
  , p_rsqrt
  , p_ceil
  , p_floor
  , p_round
  , p_trunc
  , p_frac
  , p_cinv
  , p_vectorDispatchInit
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h c_THVectorDouble_fill"
  c_fill :: Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (())

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h c_THVectorDouble_cadd"
  c_cadd :: Ptr (CDouble) -> Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (())

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h c_THVectorDouble_adds"
  c_adds :: Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (())

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h c_THVectorDouble_cmul"
  c_cmul :: Ptr (CDouble) -> Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h c_THVectorDouble_muls"
  c_muls :: Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (())

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h c_THVectorDouble_cdiv"
  c_cdiv :: Ptr (CDouble) -> Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h c_THVectorDouble_divs"
  c_divs :: Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (())

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_copy"
  c_copy :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_neg :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_neg"
  c_neg :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h c_THVectorDouble_normal_fill"
  c_normal_fill :: Ptr (CDouble) -> CLLong -> Ptr (CTHGenerator) -> CDouble -> CDouble -> IO (())

-- | c_abs :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_abs"
  c_abs :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_log :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_log"
  c_log :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_lgamma :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_lgamma"
  c_lgamma :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_digamma :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_digamma"
  c_digamma :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_trigamma :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_trigamma"
  c_trigamma :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_log1p :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_log1p"
  c_log1p :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_sigmoid :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_sigmoid"
  c_sigmoid :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_exp :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_exp"
  c_exp :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_expm1 :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_expm1"
  c_expm1 :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_erf :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_erf"
  c_erf :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_erfinv :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_erfinv"
  c_erfinv :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_cos :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_cos"
  c_cos :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_acos :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_acos"
  c_acos :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_cosh :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_cosh"
  c_cosh :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_sin :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_sin"
  c_sin :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_asin :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_asin"
  c_asin :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_sinh :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_sinh"
  c_sinh :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_tan :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_tan"
  c_tan :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_atan :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_atan"
  c_atan :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_tanh :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_tanh"
  c_tanh :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_pow :  y x c n -> void
foreign import ccall "THVector.h c_THVectorDouble_pow"
  c_pow :: Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (())

-- | c_sqrt :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_sqrt"
  c_sqrt :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_rsqrt :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_rsqrt"
  c_rsqrt :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_ceil :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_ceil"
  c_ceil :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_floor :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_floor"
  c_floor :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_round :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_round"
  c_round :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_trunc :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_trunc"
  c_trunc :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_frac :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_frac"
  c_frac :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_cinv :  y x n -> void
foreign import ccall "THVector.h c_THVectorDouble_cinv"
  c_cinv :: Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (())

-- | c_vectorDispatchInit :   -> void
foreign import ccall "THVector.h c_THVectorDouble_vectorDispatchInit"
  c_vectorDispatchInit :: IO (())

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &p_THVectorDouble_fill"
  p_fill :: FunPtr (Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (()))

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &p_THVectorDouble_cadd"
  p_cadd :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (()))

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &p_THVectorDouble_adds"
  p_adds :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (()))

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &p_THVectorDouble_cmul"
  p_cmul :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &p_THVectorDouble_muls"
  p_muls :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (()))

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &p_THVectorDouble_cdiv"
  p_cdiv :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &p_THVectorDouble_divs"
  p_divs :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (()))

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_copy"
  p_copy :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_neg"
  p_neg :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &p_THVectorDouble_normal_fill"
  p_normal_fill :: FunPtr (Ptr (CDouble) -> CLLong -> Ptr (CTHGenerator) -> CDouble -> CDouble -> IO (()))

-- | p_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_abs"
  p_abs :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_log : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_log"
  p_log :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_lgamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_lgamma"
  p_lgamma :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_digamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_digamma"
  p_digamma :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_trigamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_trigamma"
  p_trigamma :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_log1p : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_log1p"
  p_log1p :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_sigmoid : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_sigmoid"
  p_sigmoid :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_exp : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_exp"
  p_exp :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_expm1 : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_expm1"
  p_expm1 :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_erf : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_erf"
  p_erf :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_erfinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_erfinv"
  p_erfinv :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_cos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_cos"
  p_cos :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_acos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_acos"
  p_acos :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_cosh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_cosh"
  p_cosh :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_sin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_sin"
  p_sin :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_asin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_asin"
  p_asin :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_sinh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_sinh"
  p_sinh :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_tan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_tan"
  p_tan :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_atan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_atan"
  p_atan :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_tanh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_tanh"
  p_tanh :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_pow : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &p_THVectorDouble_pow"
  p_pow :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CDouble -> CPtrdiff -> IO (()))

-- | p_sqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_sqrt"
  p_sqrt :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_rsqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_rsqrt"
  p_rsqrt :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_ceil : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_ceil"
  p_ceil :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_floor : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_floor"
  p_floor :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_round : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_round"
  p_round :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_trunc : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_trunc"
  p_trunc :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_frac : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_frac"
  p_frac :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_cinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &p_THVectorDouble_cinv"
  p_cinv :: FunPtr (Ptr (CDouble) -> Ptr (CDouble) -> CPtrdiff -> IO (()))

-- | p_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &p_THVectorDouble_vectorDispatchInit"
  p_vectorDispatchInit :: FunPtr (IO (()))