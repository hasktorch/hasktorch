{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleVector (
    c_THDoubleVector_fill,
    c_THDoubleVector_cadd,
    c_THDoubleVector_adds,
    c_THDoubleVector_cmul,
    c_THDoubleVector_muls,
    c_THDoubleVector_cdiv,
    c_THDoubleVector_divs,
    c_THDoubleVector_copy,
    c_THDoubleVector_neg,
    c_THDoubleVector_abs,
    c_THDoubleVector_log,
    c_THDoubleVector_lgamma,
    c_THDoubleVector_log1p,
    c_THDoubleVector_sigmoid,
    c_THDoubleVector_exp,
    c_THDoubleVector_erf,
    c_THDoubleVector_erfinv,
    c_THDoubleVector_cos,
    c_THDoubleVector_acos,
    c_THDoubleVector_cosh,
    c_THDoubleVector_sin,
    c_THDoubleVector_asin,
    c_THDoubleVector_sinh,
    c_THDoubleVector_tan,
    c_THDoubleVector_atan,
    c_THDoubleVector_tanh,
    c_THDoubleVector_pow,
    c_THDoubleVector_sqrt,
    c_THDoubleVector_rsqrt,
    c_THDoubleVector_ceil,
    c_THDoubleVector_floor,
    c_THDoubleVector_round,
    c_THDoubleVector_trunc,
    c_THDoubleVector_frac,
    c_THDoubleVector_cinv,
    c_THDoubleVector_vectorDispatchInit,
    p_THDoubleVector_fill,
    p_THDoubleVector_cadd,
    p_THDoubleVector_adds,
    p_THDoubleVector_cmul,
    p_THDoubleVector_muls,
    p_THDoubleVector_cdiv,
    p_THDoubleVector_divs,
    p_THDoubleVector_copy,
    p_THDoubleVector_neg,
    p_THDoubleVector_abs,
    p_THDoubleVector_log,
    p_THDoubleVector_lgamma,
    p_THDoubleVector_log1p,
    p_THDoubleVector_sigmoid,
    p_THDoubleVector_exp,
    p_THDoubleVector_erf,
    p_THDoubleVector_erfinv,
    p_THDoubleVector_cos,
    p_THDoubleVector_acos,
    p_THDoubleVector_cosh,
    p_THDoubleVector_sin,
    p_THDoubleVector_asin,
    p_THDoubleVector_sinh,
    p_THDoubleVector_tan,
    p_THDoubleVector_atan,
    p_THDoubleVector_tanh,
    p_THDoubleVector_pow,
    p_THDoubleVector_sqrt,
    p_THDoubleVector_rsqrt,
    p_THDoubleVector_ceil,
    p_THDoubleVector_floor,
    p_THDoubleVector_round,
    p_THDoubleVector_trunc,
    p_THDoubleVector_frac,
    p_THDoubleVector_cinv,
    p_THDoubleVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THDoubleVector_fill : x c n -> void
foreign import ccall "THVector.h THDoubleVector_fill"
  c_THDoubleVector_fill :: Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THDoubleVector_cadd"
  c_THDoubleVector_cadd :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_adds : y x c n -> void
foreign import ccall "THVector.h THDoubleVector_adds"
  c_THDoubleVector_adds :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cmul : z x y n -> void
foreign import ccall "THVector.h THDoubleVector_cmul"
  c_THDoubleVector_cmul :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_muls : y x c n -> void
foreign import ccall "THVector.h THDoubleVector_muls"
  c_THDoubleVector_muls :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THDoubleVector_cdiv"
  c_THDoubleVector_cdiv :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_divs : y x c n -> void
foreign import ccall "THVector.h THDoubleVector_divs"
  c_THDoubleVector_divs :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_copy : y x n -> void
foreign import ccall "THVector.h THDoubleVector_copy"
  c_THDoubleVector_copy :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_neg : y x n -> void
foreign import ccall "THVector.h THDoubleVector_neg"
  c_THDoubleVector_neg :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_abs : y x n -> void
foreign import ccall "THVector.h THDoubleVector_abs"
  c_THDoubleVector_abs :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_log : y x n -> void
foreign import ccall "THVector.h THDoubleVector_log"
  c_THDoubleVector_log :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_lgamma : y x n -> void
foreign import ccall "THVector.h THDoubleVector_lgamma"
  c_THDoubleVector_lgamma :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_log1p : y x n -> void
foreign import ccall "THVector.h THDoubleVector_log1p"
  c_THDoubleVector_log1p :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_sigmoid : y x n -> void
foreign import ccall "THVector.h THDoubleVector_sigmoid"
  c_THDoubleVector_sigmoid :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_exp : y x n -> void
foreign import ccall "THVector.h THDoubleVector_exp"
  c_THDoubleVector_exp :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_erf : y x n -> void
foreign import ccall "THVector.h THDoubleVector_erf"
  c_THDoubleVector_erf :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_erfinv : y x n -> void
foreign import ccall "THVector.h THDoubleVector_erfinv"
  c_THDoubleVector_erfinv :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cos : y x n -> void
foreign import ccall "THVector.h THDoubleVector_cos"
  c_THDoubleVector_cos :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_acos : y x n -> void
foreign import ccall "THVector.h THDoubleVector_acos"
  c_THDoubleVector_acos :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cosh : y x n -> void
foreign import ccall "THVector.h THDoubleVector_cosh"
  c_THDoubleVector_cosh :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_sin : y x n -> void
foreign import ccall "THVector.h THDoubleVector_sin"
  c_THDoubleVector_sin :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_asin : y x n -> void
foreign import ccall "THVector.h THDoubleVector_asin"
  c_THDoubleVector_asin :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_sinh : y x n -> void
foreign import ccall "THVector.h THDoubleVector_sinh"
  c_THDoubleVector_sinh :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_tan : y x n -> void
foreign import ccall "THVector.h THDoubleVector_tan"
  c_THDoubleVector_tan :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_atan : y x n -> void
foreign import ccall "THVector.h THDoubleVector_atan"
  c_THDoubleVector_atan :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_tanh : y x n -> void
foreign import ccall "THVector.h THDoubleVector_tanh"
  c_THDoubleVector_tanh :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_pow : y x c n -> void
foreign import ccall "THVector.h THDoubleVector_pow"
  c_THDoubleVector_pow :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_sqrt : y x n -> void
foreign import ccall "THVector.h THDoubleVector_sqrt"
  c_THDoubleVector_sqrt :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_rsqrt : y x n -> void
foreign import ccall "THVector.h THDoubleVector_rsqrt"
  c_THDoubleVector_rsqrt :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_ceil : y x n -> void
foreign import ccall "THVector.h THDoubleVector_ceil"
  c_THDoubleVector_ceil :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_floor : y x n -> void
foreign import ccall "THVector.h THDoubleVector_floor"
  c_THDoubleVector_floor :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_round : y x n -> void
foreign import ccall "THVector.h THDoubleVector_round"
  c_THDoubleVector_round :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_trunc : y x n -> void
foreign import ccall "THVector.h THDoubleVector_trunc"
  c_THDoubleVector_trunc :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_frac : y x n -> void
foreign import ccall "THVector.h THDoubleVector_frac"
  c_THDoubleVector_frac :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cinv : y x n -> void
foreign import ccall "THVector.h THDoubleVector_cinv"
  c_THDoubleVector_cinv :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THDoubleVector_vectorDispatchInit"
  c_THDoubleVector_vectorDispatchInit :: IO ()

-- |p_THDoubleVector_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THDoubleVector_fill"
  p_THDoubleVector_fill :: FunPtr (Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THDoubleVector_cadd"
  p_THDoubleVector_cadd :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THDoubleVector_adds"
  p_THDoubleVector_adds :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THDoubleVector_cmul"
  p_THDoubleVector_cmul :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THDoubleVector_muls"
  p_THDoubleVector_muls :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THDoubleVector_cdiv"
  p_THDoubleVector_cdiv :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THDoubleVector_divs"
  p_THDoubleVector_divs :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_copy"
  p_THDoubleVector_copy :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_neg"
  p_THDoubleVector_neg :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_abs"
  p_THDoubleVector_abs :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_log : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_log"
  p_THDoubleVector_log :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_lgamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_lgamma"
  p_THDoubleVector_lgamma :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_log1p : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_log1p"
  p_THDoubleVector_log1p :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_sigmoid : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_sigmoid"
  p_THDoubleVector_sigmoid :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_exp : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_exp"
  p_THDoubleVector_exp :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_erf : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_erf"
  p_THDoubleVector_erf :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_erfinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_erfinv"
  p_THDoubleVector_erfinv :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_cos"
  p_THDoubleVector_cos :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_acos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_acos"
  p_THDoubleVector_acos :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cosh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_cosh"
  p_THDoubleVector_cosh :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_sin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_sin"
  p_THDoubleVector_sin :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_asin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_asin"
  p_THDoubleVector_asin :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_sinh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_sinh"
  p_THDoubleVector_sinh :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_tan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_tan"
  p_THDoubleVector_tan :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_atan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_atan"
  p_THDoubleVector_atan :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_tanh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_tanh"
  p_THDoubleVector_tanh :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_pow : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THDoubleVector_pow"
  p_THDoubleVector_pow :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_sqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_sqrt"
  p_THDoubleVector_sqrt :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_rsqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_rsqrt"
  p_THDoubleVector_rsqrt :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_ceil : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_ceil"
  p_THDoubleVector_ceil :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_floor : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_floor"
  p_THDoubleVector_floor :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_round : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_round"
  p_THDoubleVector_round :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_trunc : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_trunc"
  p_THDoubleVector_trunc :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_frac : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_frac"
  p_THDoubleVector_frac :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THDoubleVector_cinv"
  p_THDoubleVector_cinv :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THDoubleVector_vectorDispatchInit"
  p_THDoubleVector_vectorDispatchInit :: FunPtr (IO ())