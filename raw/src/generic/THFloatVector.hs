{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatVector (
    c_THFloatVector_fill,
    c_THFloatVector_cadd,
    c_THFloatVector_adds,
    c_THFloatVector_cmul,
    c_THFloatVector_muls,
    c_THFloatVector_cdiv,
    c_THFloatVector_divs,
    c_THFloatVector_copy,
    c_THFloatVector_abs,
    c_THFloatVector_log,
    c_THFloatVector_lgamma,
    c_THFloatVector_log1p,
    c_THFloatVector_sigmoid,
    c_THFloatVector_exp,
    c_THFloatVector_erf,
    c_THFloatVector_erfinv,
    c_THFloatVector_cos,
    c_THFloatVector_acos,
    c_THFloatVector_cosh,
    c_THFloatVector_sin,
    c_THFloatVector_asin,
    c_THFloatVector_sinh,
    c_THFloatVector_tan,
    c_THFloatVector_atan,
    c_THFloatVector_tanh,
    c_THFloatVector_pow,
    c_THFloatVector_sqrt,
    c_THFloatVector_rsqrt,
    c_THFloatVector_ceil,
    c_THFloatVector_floor,
    c_THFloatVector_round,
    c_THFloatVector_trunc,
    c_THFloatVector_frac,
    c_THFloatVector_cinv,
    c_THFloatVector_neg,
    c_THFloatVector_vectorDispatchInit,
    p_THFloatVector_fill,
    p_THFloatVector_cadd,
    p_THFloatVector_adds,
    p_THFloatVector_cmul,
    p_THFloatVector_muls,
    p_THFloatVector_cdiv,
    p_THFloatVector_divs,
    p_THFloatVector_copy,
    p_THFloatVector_abs,
    p_THFloatVector_log,
    p_THFloatVector_lgamma,
    p_THFloatVector_log1p,
    p_THFloatVector_sigmoid,
    p_THFloatVector_exp,
    p_THFloatVector_erf,
    p_THFloatVector_erfinv,
    p_THFloatVector_cos,
    p_THFloatVector_acos,
    p_THFloatVector_cosh,
    p_THFloatVector_sin,
    p_THFloatVector_asin,
    p_THFloatVector_sinh,
    p_THFloatVector_tan,
    p_THFloatVector_atan,
    p_THFloatVector_tanh,
    p_THFloatVector_pow,
    p_THFloatVector_sqrt,
    p_THFloatVector_rsqrt,
    p_THFloatVector_ceil,
    p_THFloatVector_floor,
    p_THFloatVector_round,
    p_THFloatVector_trunc,
    p_THFloatVector_frac,
    p_THFloatVector_cinv,
    p_THFloatVector_neg,
    p_THFloatVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THFloatVector_fill : x c n -> void
foreign import ccall "THVector.h THFloatVector_fill"
  c_THFloatVector_fill :: Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THFloatVector_cadd"
  c_THFloatVector_cadd :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_adds : y x c n -> void
foreign import ccall "THVector.h THFloatVector_adds"
  c_THFloatVector_adds :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cmul : z x y n -> void
foreign import ccall "THVector.h THFloatVector_cmul"
  c_THFloatVector_cmul :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_muls : y x c n -> void
foreign import ccall "THVector.h THFloatVector_muls"
  c_THFloatVector_muls :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THFloatVector_cdiv"
  c_THFloatVector_cdiv :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_divs : y x c n -> void
foreign import ccall "THVector.h THFloatVector_divs"
  c_THFloatVector_divs :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_copy : y x n -> void
foreign import ccall "THVector.h THFloatVector_copy"
  c_THFloatVector_copy :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_abs : y x n -> void
foreign import ccall "THVector.h THFloatVector_abs"
  c_THFloatVector_abs :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_log : y x n -> void
foreign import ccall "THVector.h THFloatVector_log"
  c_THFloatVector_log :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_lgamma : y x n -> void
foreign import ccall "THVector.h THFloatVector_lgamma"
  c_THFloatVector_lgamma :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_log1p : y x n -> void
foreign import ccall "THVector.h THFloatVector_log1p"
  c_THFloatVector_log1p :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_sigmoid : y x n -> void
foreign import ccall "THVector.h THFloatVector_sigmoid"
  c_THFloatVector_sigmoid :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_exp : y x n -> void
foreign import ccall "THVector.h THFloatVector_exp"
  c_THFloatVector_exp :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_erf : y x n -> void
foreign import ccall "THVector.h THFloatVector_erf"
  c_THFloatVector_erf :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_erfinv : y x n -> void
foreign import ccall "THVector.h THFloatVector_erfinv"
  c_THFloatVector_erfinv :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cos : y x n -> void
foreign import ccall "THVector.h THFloatVector_cos"
  c_THFloatVector_cos :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_acos : y x n -> void
foreign import ccall "THVector.h THFloatVector_acos"
  c_THFloatVector_acos :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cosh : y x n -> void
foreign import ccall "THVector.h THFloatVector_cosh"
  c_THFloatVector_cosh :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_sin : y x n -> void
foreign import ccall "THVector.h THFloatVector_sin"
  c_THFloatVector_sin :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_asin : y x n -> void
foreign import ccall "THVector.h THFloatVector_asin"
  c_THFloatVector_asin :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_sinh : y x n -> void
foreign import ccall "THVector.h THFloatVector_sinh"
  c_THFloatVector_sinh :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_tan : y x n -> void
foreign import ccall "THVector.h THFloatVector_tan"
  c_THFloatVector_tan :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_atan : y x n -> void
foreign import ccall "THVector.h THFloatVector_atan"
  c_THFloatVector_atan :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_tanh : y x n -> void
foreign import ccall "THVector.h THFloatVector_tanh"
  c_THFloatVector_tanh :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_pow : y x c n -> void
foreign import ccall "THVector.h THFloatVector_pow"
  c_THFloatVector_pow :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_sqrt : y x n -> void
foreign import ccall "THVector.h THFloatVector_sqrt"
  c_THFloatVector_sqrt :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_rsqrt : y x n -> void
foreign import ccall "THVector.h THFloatVector_rsqrt"
  c_THFloatVector_rsqrt :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_ceil : y x n -> void
foreign import ccall "THVector.h THFloatVector_ceil"
  c_THFloatVector_ceil :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_floor : y x n -> void
foreign import ccall "THVector.h THFloatVector_floor"
  c_THFloatVector_floor :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_round : y x n -> void
foreign import ccall "THVector.h THFloatVector_round"
  c_THFloatVector_round :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_trunc : y x n -> void
foreign import ccall "THVector.h THFloatVector_trunc"
  c_THFloatVector_trunc :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_frac : y x n -> void
foreign import ccall "THVector.h THFloatVector_frac"
  c_THFloatVector_frac :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cinv : y x n -> void
foreign import ccall "THVector.h THFloatVector_cinv"
  c_THFloatVector_cinv :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_neg : y x n -> void
foreign import ccall "THVector.h THFloatVector_neg"
  c_THFloatVector_neg :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THFloatVector_vectorDispatchInit"
  c_THFloatVector_vectorDispatchInit :: IO ()

-- |p_THFloatVector_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THFloatVector_fill"
  p_THFloatVector_fill :: FunPtr (Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THFloatVector_cadd"
  p_THFloatVector_cadd :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THFloatVector_adds"
  p_THFloatVector_adds :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THFloatVector_cmul"
  p_THFloatVector_cmul :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THFloatVector_muls"
  p_THFloatVector_muls :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THFloatVector_cdiv"
  p_THFloatVector_cdiv :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THFloatVector_divs"
  p_THFloatVector_divs :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_copy"
  p_THFloatVector_copy :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_abs"
  p_THFloatVector_abs :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_log : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_log"
  p_THFloatVector_log :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_lgamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_lgamma"
  p_THFloatVector_lgamma :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_log1p : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_log1p"
  p_THFloatVector_log1p :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_sigmoid : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_sigmoid"
  p_THFloatVector_sigmoid :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_exp : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_exp"
  p_THFloatVector_exp :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_erf : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_erf"
  p_THFloatVector_erf :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_erfinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_erfinv"
  p_THFloatVector_erfinv :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_cos"
  p_THFloatVector_cos :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_acos : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_acos"
  p_THFloatVector_acos :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cosh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_cosh"
  p_THFloatVector_cosh :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_sin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_sin"
  p_THFloatVector_sin :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_asin : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_asin"
  p_THFloatVector_asin :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_sinh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_sinh"
  p_THFloatVector_sinh :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_tan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_tan"
  p_THFloatVector_tan :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_atan : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_atan"
  p_THFloatVector_atan :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_tanh : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_tanh"
  p_THFloatVector_tanh :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_pow : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THFloatVector_pow"
  p_THFloatVector_pow :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_sqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_sqrt"
  p_THFloatVector_sqrt :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_rsqrt : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_rsqrt"
  p_THFloatVector_rsqrt :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_ceil : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_ceil"
  p_THFloatVector_ceil :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_floor : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_floor"
  p_THFloatVector_floor :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_round : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_round"
  p_THFloatVector_round :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_trunc : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_trunc"
  p_THFloatVector_trunc :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_frac : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_frac"
  p_THFloatVector_frac :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_cinv"
  p_THFloatVector_cinv :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THFloatVector_neg"
  p_THFloatVector_neg :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THFloatVector_vectorDispatchInit"
  p_THFloatVector_vectorDispatchInit :: FunPtr (IO ())