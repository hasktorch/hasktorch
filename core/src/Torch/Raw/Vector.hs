{-# LANGUAGE TypeSynonymInstances #-}

module Torch.Raw.Vector
  ( THVector(..)
  , module X
  ) where

import Torch.Raw.Internal as X

import qualified THFloatVector as T
import qualified THDoubleVector as T
import qualified THByteVector as T
import qualified THIntVector as T
import qualified THShortVector as T
import qualified THLongVector as T
-- import qualified THHalfVector as T

-- CDouble
class THVector t where
  c_fill :: Ptr t -> t -> CPtrdiff -> IO ()
  c_cadd :: Ptr t -> Ptr t -> Ptr t -> t -> CPtrdiff -> IO ()
  c_adds :: Ptr t -> Ptr t -> t -> CPtrdiff -> IO ()
  c_cmul :: Ptr t -> Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_muls :: Ptr t -> Ptr t -> t -> CPtrdiff -> IO ()
  c_cdiv :: Ptr t -> Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_divs :: Ptr t -> Ptr t -> t -> CPtrdiff -> IO ()
  c_copy :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_neg :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_abs :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_log :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_lgamma :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_log1p :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_sigmoid :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_exp :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_erf :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_erfinv :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_cos :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_acos :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_cosh :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_sin :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_asin :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_sinh :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_tan :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_atan :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_tanh :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_pow :: Ptr t -> Ptr t -> t -> CPtrdiff -> IO ()
  c_sqrt :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_rsqrt :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_ceil :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_floor :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_round :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_trunc :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_frac :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  c_cinv :: Ptr t -> Ptr t -> CPtrdiff -> IO ()
  -- c_vectorDispatchInit :: IO ()
  p_fill :: FunPtr (Ptr t -> t -> CPtrdiff -> IO ())
  p_cadd :: FunPtr (Ptr t -> Ptr t -> Ptr t -> t -> CPtrdiff -> IO ())
  p_adds :: FunPtr (Ptr t -> Ptr t -> t -> CPtrdiff -> IO ())
  p_cmul :: FunPtr (Ptr t -> Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_muls :: FunPtr (Ptr t -> Ptr t -> t -> CPtrdiff -> IO ())
  p_cdiv :: FunPtr (Ptr t -> Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_divs :: FunPtr (Ptr t -> Ptr t -> t -> CPtrdiff -> IO ())
  p_copy :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_neg :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_abs :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_log :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_lgamma :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_log1p :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_sigmoid :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_exp :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_erf :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_erfinv :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_cos :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_acos :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_cosh :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_sin :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_asin :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_sinh :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_tan :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_atan :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_tanh :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_pow :: FunPtr (Ptr t -> Ptr t -> t -> CPtrdiff -> IO ())
  p_sqrt :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_rsqrt :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_ceil :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_floor :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_round :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_trunc :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_frac :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  p_cinv :: FunPtr (Ptr t -> Ptr t -> CPtrdiff -> IO ())
  -- p_vectorDispatchInit :: FunPtr (IO ())

instance THVector CTHFloatVector where
  c_fill = T.c_THFloatVector_fill
  c_cadd = T.c_THFloatVector_cadd
  c_adds = T.c_THFloatVector_adds
  c_cmul = T.c_THFloatVector_cmul
  c_muls = T.c_THFloatVector_muls
  c_cdiv = T.c_THFloatVector_cdiv
  c_divs = T.c_THFloatVector_divs
  c_copy = T.c_THFloatVector_copy
  c_neg = T.c_THFloatVector_neg
  c_abs = T.c_THFloatVector_abs
  c_log = T.c_THFloatVector_log
  c_lgamma = T.c_THFloatVector_lgamma
  c_log1p = T.c_THFloatVector_log1p
  c_sigmoid = T.c_THFloatVector_sigmoid
  c_exp = T.c_THFloatVector_exp
  c_erf = T.c_THFloatVector_erf
  c_erfinv = T.c_THFloatVector_erfinv
  c_cos = T.c_THFloatVector_cos
  c_acos = T.c_THFloatVector_acos
  c_cosh = T.c_THFloatVector_cosh
  c_sin = T.c_THFloatVector_sin
  c_asin = T.c_THFloatVector_asin
  c_sinh = T.c_THFloatVector_sinh
  c_tan = T.c_THFloatVector_tan
  c_atan = T.c_THFloatVector_atan
  c_tanh = T.c_THFloatVector_tanh
  c_pow = T.c_THFloatVector_pow
  c_sqrt = T.c_THFloatVector_sqrt
  c_rsqrt = T.c_THFloatVector_rsqrt
  c_ceil = T.c_THFloatVector_ceil
  c_floor = T.c_THFloatVector_floor
  c_round = T.c_THFloatVector_round
  c_trunc = T.c_THFloatVector_trunc
  c_frac = T.c_THFloatVector_frac
  c_cinv = T.c_THFloatVector_cinv
  -- c_vec = T.THFloatVector_--
  p_fill = T.p_THFloatVector_fill
  p_cadd = T.p_THFloatVector_cadd
  p_adds = T.p_THFloatVector_adds
  p_cmul = T.p_THFloatVector_cmul
  p_muls = T.p_THFloatVector_muls
  p_cdiv = T.p_THFloatVector_cdiv
  p_divs = T.p_THFloatVector_divs
  p_copy = T.p_THFloatVector_copy
  p_neg = T.p_THFloatVector_neg
  p_abs = T.p_THFloatVector_abs
  p_log = T.p_THFloatVector_log
  p_lgamma = T.p_THFloatVector_lgamma
  p_log1p = T.p_THFloatVector_log1p
  p_sigmoid = T.p_THFloatVector_sigmoid
  p_exp = T.p_THFloatVector_exp
  p_erf = T.p_THFloatVector_erf
  p_erfinv = T.p_THFloatVector_erfinv
  p_cos = T.p_THFloatVector_cos
  p_acos = T.p_THFloatVector_acos
  p_cosh = T.p_THFloatVector_cosh
  p_sin = T.p_THFloatVector_sin
  p_asin = T.p_THFloatVector_asin
  p_sinh = T.p_THFloatVector_sinh
  p_tan = T.p_THFloatVector_tan
  p_atan = T.p_THFloatVector_atan
  p_tanh = T.p_THFloatVector_tanh
  p_pow = T.p_THFloatVector_pow
  p_sqrt = T.p_THFloatVector_sqrt
  p_rsqrt = T.p_THFloatVector_rsqrt
  p_ceil = T.p_THFloatVector_ceil
  p_floor = T.p_THFloatVector_floor
  p_round = T.p_THFloatVector_round
  p_trunc = T.p_THFloatVector_trunc
  p_frac = T.p_THFloatVector_frac
  p_cinv = T.p_THFloatVector_cinv

-- instance THVector CTHDoubleVector where
--   c_fill = T.c_THDoubleVector_fill
--   c_cadd = T.c_THDoubleVector_cadd
--   c_adds = T.c_THDoubleVector_adds
--   c_cmul = T.c_THDoubleVector_cmul
--   c_muls = T.c_THDoubleVector_muls
--   c_cdiv = T.c_THDoubleVector_cdiv
--   c_divs = T.c_THDoubleVector_divs
--   c_copy = T.c_THDoubleVector_copy
--   c_neg = T.c_THDoubleVector_neg
--   c_abs = T.c_THDoubleVector_abs
--   c_log = T.c_THDoubleVector_log
--   c_lgamma = T.c_THDoubleVector_lgamma
--   c_log1p = T.c_THDoubleVector_log1p
--   c_sigmoid = T.c_THDoubleVector_sigmoid
--   c_exp = T.c_THDoubleVector_exp
--   c_erf = T.c_THDoubleVector_erf
--   c_erfinv = T.c_THDoubleVector_erfinv
--   c_cos = T.c_THDoubleVector_cos
--   c_acos = T.c_THDoubleVector_acos
--   c_cosh = T.c_THDoubleVector_cosh
--   c_sin = T.c_THDoubleVector_sin
--   c_asin = T.c_THDoubleVector_asin
--   c_sinh = T.c_THDoubleVector_sinh
--   c_tan = T.c_THDoubleVector_tan
--   c_atan = T.c_THDoubleVector_atan
--   c_tanh = T.c_THDoubleVector_tanh
--   c_pow = T.c_THDoubleVector_pow
--   c_sqrt = T.c_THDoubleVector_sqrt
--   c_rsqrt = T.c_THDoubleVector_rsqrt
--   c_ceil = T.c_THDoubleVector_ceil
--   c_floor = T.c_THDoubleVector_floor
--   c_round = T.c_THDoubleVector_round
--   c_trunc = T.c_THDoubleVector_trunc
--   c_frac = T.c_THDoubleVector_frac
--   c_cinv = T.c_THDoubleVector_cinv
--   -- c_vec = T.THDoubleVector_--
--   p_fill = T.p_THDoubleVector_fill
--   p_cadd = T.p_THDoubleVector_cadd
--   p_adds = T.p_THDoubleVector_adds
--   p_cmul = T.p_THDoubleVector_cmul
--   p_muls = T.p_THDoubleVector_muls
--   p_cdiv = T.p_THDoubleVector_cdiv
--   p_divs = T.p_THDoubleVector_divs
--   p_copy = T.p_THDoubleVector_copy
--   p_neg = T.p_THDoubleVector_neg
--   p_abs = T.p_THDoubleVector_abs
--   p_log = T.p_THDoubleVector_log
--   p_lgamma = T.p_THDoubleVector_lgamma
--   p_log1p = T.p_THDoubleVector_log1p
--   p_sigmoid = T.p_THDoubleVector_sigmoid
--   p_exp = T.p_THDoubleVector_exp
--   p_erf = T.p_THDoubleVector_erf
--   p_erfinv = T.p_THDoubleVector_erfinv
--   p_cos = T.p_THDoubleVector_cos
--   p_acos = T.p_THDoubleVector_acos
--   p_cosh = T.p_THDoubleVector_cosh
--   p_sin = T.p_THDoubleVector_sin
--   p_asin = T.p_THDoubleVector_asin
--   p_sinh = T.p_THDoubleVector_sinh
--   p_tan = T.p_THDoubleVector_tan
--   p_atan = T.p_THDoubleVector_atan
--   p_tanh = T.p_THDoubleVector_tanh
--   p_pow = T.p_THDoubleVector_pow
--   p_sqrt = T.p_THDoubleVector_sqrt
--   p_rsqrt = T.p_THDoubleVector_rsqrt
--   p_ceil = T.p_THDoubleVector_ceil
--   p_floor = T.p_THDoubleVector_floor
--   p_round = T.p_THDoubleVector_round
--   p_trunc = T.p_THDoubleVector_trunc
--   p_frac = T.p_THDoubleVector_frac

-- instance THVector CTHByteVector where
--   c_fill = T.c_THByteVector_fill
--   c_cadd = T.c_THByteVector_cadd
--   c_adds = T.c_THByteVector_adds
--   c_cmul = T.c_THByteVector_cmul
--   c_muls = T.c_THByteVector_muls
--   c_cdiv = T.c_THByteVector_cdiv
--   c_divs = T.c_THByteVector_divs
--   c_copy = T.c_THByteVector_copy
--   c_neg = T.c_THByteVector_neg
--   c_abs = T.c_THByteVector_abs
--   c_log = T.c_THByteVector_log
--   c_lgamma = T.c_THByteVector_lgamma
--   c_log1p = T.c_THByteVector_log1p
--   c_sigmoid = T.c_THByteVector_sigmoid
--   c_exp = T.c_THByteVector_exp
--   c_erf = T.c_THByteVector_erf
--   c_erfinv = T.c_THByteVector_erfinv
--   c_cos = T.c_THByteVector_cos
--   c_acos = T.c_THByteVector_acos
--   c_cosh = T.c_THByteVector_cosh
--   c_sin = T.c_THByteVector_sin
--   c_asin = T.c_THByteVector_asin
--   c_sinh = T.c_THByteVector_sinh
--   c_tan = T.c_THByteVector_tan
--   c_atan = T.c_THByteVector_atan
--   c_tanh = T.c_THByteVector_tanh
--   c_pow = T.c_THByteVector_pow
--   c_sqrt = T.c_THByteVector_sqrt
--   c_rsqrt = T.c_THByteVector_rsqrt
--   c_ceil = T.c_THByteVector_ceil
--   c_floor = T.c_THByteVector_floor
--   c_round = T.c_THByteVector_round
--   c_trunc = T.c_THByteVector_trunc
--   c_frac = T.c_THByteVector_frac
--   c_cinv = T.c_THByteVector_cinv
--   -- c_vec = T.THByteVector_--
--   p_fill = T.p_THByteVector_fill
--   p_cadd = T.p_THByteVector_cadd
--   p_adds = T.p_THByteVector_adds
--   p_cmul = T.p_THByteVector_cmul
--   p_muls = T.p_THByteVector_muls
--   p_cdiv = T.p_THByteVector_cdiv
--   p_divs = T.p_THByteVector_divs
--   p_copy = T.p_THByteVector_copy
--   p_neg = T.p_THByteVector_neg
--   p_abs = T.p_THByteVector_abs
--   p_log = T.p_THByteVector_log
--   p_lgamma = T.p_THByteVector_lgamma
--   p_log1p = T.p_THByteVector_log1p
--   p_sigmoid = T.p_THByteVector_sigmoid
--   p_exp = T.p_THByteVector_exp
--   p_erf = T.p_THByteVector_erf
--   p_erfinv = T.p_THByteVector_erfinv
--   p_cos = T.p_THByteVector_cos
--   p_acos = T.p_THByteVector_acos
--   p_cosh = T.p_THByteVector_cosh
--   p_sin = T.p_THByteVector_sin
--   p_asin = T.p_THByteVector_asin
--   p_sinh = T.p_THByteVector_sinh
--   p_tan = T.p_THByteVector_tan
--   p_atan = T.p_THByteVector_atan
--   p_tanh = T.p_THByteVector_tanh
--   p_pow = T.p_THByteVector_pow
--   p_sqrt = T.p_THByteVector_sqrt
--   p_rsqrt = T.p_THByteVector_rsqrt
--   p_ceil = T.p_THByteVector_ceil
--   p_floor = T.p_THByteVector_floor
--   p_round = T.p_THByteVector_round
--   p_trunc = T.p_THByteVector_trunc
--   p_frac = T.p_THByteVector_frac
--   p_cinv = T.p_THByteVector_cinv
--   p_cinv = T.p_THByteVector_cinv

-- instance THVector CTHIntVector where
--   c_fill = T.c_THIntVector_fill
--   c_cadd = T.c_THIntVector_cadd
--   c_adds = T.c_THIntVector_adds
--   c_cmul = T.c_THIntVector_cmul
--   c_muls = T.c_THIntVector_muls
--   c_cdiv = T.c_THIntVector_cdiv
--   c_divs = T.c_THIntVector_divs
--   c_copy = T.c_THIntVector_copy
--   c_neg = T.c_THIntVector_neg
--   c_abs = T.c_THIntVector_abs
--   c_log = T.c_THIntVector_log
--   c_lgamma = T.c_THIntVector_lgamma
--   c_log1p = T.c_THIntVector_log1p
--   c_sigmoid = T.c_THIntVector_sigmoid
--   c_exp = T.c_THIntVector_exp
--   c_erf = T.c_THIntVector_erf
--   c_erfinv = T.c_THIntVector_erfinv
--   c_cos = T.c_THIntVector_cos
--   c_acos = T.c_THIntVector_acos
--   c_cosh = T.c_THIntVector_cosh
--   c_sin = T.c_THIntVector_sin
--   c_asin = T.c_THIntVector_asin
--   c_sinh = T.c_THIntVector_sinh
--   c_tan = T.c_THIntVector_tan
--   c_atan = T.c_THIntVector_atan
--   c_tanh = T.c_THIntVector_tanh
--   c_pow = T.c_THIntVector_pow
--   c_sqrt = T.c_THIntVector_sqrt
--   c_rsqrt = T.c_THIntVector_rsqrt
--   c_ceil = T.c_THIntVector_ceil
--   c_floor = T.c_THIntVector_floor
--   c_round = T.c_THIntVector_round
--   c_trunc = T.c_THIntVector_trunc
--   c_frac = T.c_THIntVector_frac
--   c_cinv = T.c_THIntVector_cinv
--   -- c_vec = T.THIntVector_--
--   p_fill = T.p_THIntVector_fill
--   p_cadd = T.p_THIntVector_cadd
--   p_adds = T.p_THIntVector_adds
--   p_cmul = T.p_THIntVector_cmul
--   p_muls = T.p_THIntVector_muls
--   p_cdiv = T.p_THIntVector_cdiv
--   p_divs = T.p_THIntVector_divs
--   p_copy = T.p_THIntVector_copy
--   p_neg = T.p_THIntVector_neg
--   p_abs = T.p_THIntVector_abs
--   p_log = T.p_THIntVector_log
--   p_lgamma = T.p_THIntVector_lgamma
--   p_log1p = T.p_THIntVector_log1p
--   p_sigmoid = T.p_THIntVector_sigmoid
--   p_exp = T.p_THIntVector_exp
--   p_erf = T.p_THIntVector_erf
--   p_erfinv = T.p_THIntVector_erfinv
--   p_cos = T.p_THIntVector_cos
--   p_acos = T.p_THIntVector_acos
--   p_cosh = T.p_THIntVector_cosh
--   p_sin = T.p_THIntVector_sin
--   p_asin = T.p_THIntVector_asin
--   p_sinh = T.p_THIntVector_sinh
--   p_tan = T.p_THIntVector_tan
--   p_atan = T.p_THIntVector_atan
--   p_tanh = T.p_THIntVector_tanh
--   p_pow = T.p_THIntVector_pow
--   p_sqrt = T.p_THIntVector_sqrt
--   p_rsqrt = T.p_THIntVector_rsqrt
--   p_ceil = T.p_THIntVector_ceil
--   p_floor = T.p_THIntVector_floor
--   p_round = T.p_THIntVector_round
--   p_trunc = T.p_THIntVector_trunc
--   p_frac = T.p_THIntVector_frac
--   p_cinv = T.p_THIntVector_cinv
--   p_cinv = T.p_THIntVector_cinv

-- instance THVector CTHShortVector where
--   c_fill = T.c_THShortVector_fill
--   c_cadd = T.c_THShortVector_cadd
--   c_adds = T.c_THShortVector_adds
--   c_cmul = T.c_THShortVector_cmul
--   c_muls = T.c_THShortVector_muls
--   c_cdiv = T.c_THShortVector_cdiv
--   c_divs = T.c_THShortVector_divs
--   c_copy = T.c_THShortVector_copy
--   c_neg = T.c_THShortVector_neg
--   c_abs = T.c_THShortVector_abs
--   c_log = T.c_THShortVector_log
--   c_lgamma = T.c_THShortVector_lgamma
--   c_log1p = T.c_THShortVector_log1p
--   c_sigmoid = T.c_THShortVector_sigmoid
--   c_exp = T.c_THShortVector_exp
--   c_erf = T.c_THShortVector_erf
--   c_erfinv = T.c_THShortVector_erfinv
--   c_cos = T.c_THShortVector_cos
--   c_acos = T.c_THShortVector_acos
--   c_cosh = T.c_THShortVector_cosh
--   c_sin = T.c_THShortVector_sin
--   c_asin = T.c_THShortVector_asin
--   c_sinh = T.c_THShortVector_sinh
--   c_tan = T.c_THShortVector_tan
--   c_atan = T.c_THShortVector_atan
--   c_tanh = T.c_THShortVector_tanh
--   c_pow = T.c_THShortVector_pow
--   c_sqrt = T.c_THShortVector_sqrt
--   c_rsqrt = T.c_THShortVector_rsqrt
--   c_ceil = T.c_THShortVector_ceil
--   c_floor = T.c_THShortVector_floor
--   c_round = T.c_THShortVector_round
--   c_trunc = T.c_THShortVector_trunc
--   c_frac = T.c_THShortVector_frac
--   c_cinv = T.c_THShortVector_cinv
--   -- c_vec = T.THShortVector_--
--   p_fill = T.p_THShortVector_fill
--   p_cadd = T.p_THShortVector_cadd
--   p_adds = T.p_THShortVector_adds
--   p_cmul = T.p_THShortVector_cmul
--   p_muls = T.p_THShortVector_muls
--   p_cdiv = T.p_THShortVector_cdiv
--   p_divs = T.p_THShortVector_divs
--   p_copy = T.p_THShortVector_copy
--   p_neg = T.p_THShortVector_neg
--   p_abs = T.p_THShortVector_abs
--   p_log = T.p_THShortVector_log
--   p_lgamma = T.p_THShortVector_lgamma
--   p_log1p = T.p_THShortVector_log1p
--   p_sigmoid = T.p_THShortVector_sigmoid
--   p_exp = T.p_THShortVector_exp
--   p_erf = T.p_THShortVector_erf
--   p_erfinv = T.p_THShortVector_erfinv
--   p_cos = T.p_THShortVector_cos
--   p_acos = T.p_THShortVector_acos
--   p_cosh = T.p_THShortVector_cosh
--   p_sin = T.p_THShortVector_sin
--   p_asin = T.p_THShortVector_asin
--   p_sinh = T.p_THShortVector_sinh
--   p_tan = T.p_THShortVector_tan
--   p_atan = T.p_THShortVector_atan
--   p_tanh = T.p_THShortVector_tanh
--   p_pow = T.p_THShortVector_pow
--   p_sqrt = T.p_THShortVector_sqrt
--   p_rsqrt = T.p_THShortVector_rsqrt
--   p_ceil = T.p_THShortVector_ceil
--   p_floor = T.p_THShortVector_floor
--   p_round = T.p_THShortVector_round
--   p_trunc = T.p_THShortVector_trunc
--   p_frac = T.p_THShortVector_frac
--   p_cinv = T.p_THShortVector_cinv
--   p_cinv = T.p_THShortVector_cinv

-- instance THVector CTHLongVector where
--   c_fill = T.c_THLongVector_fill
--   c_cadd = T.c_THLongVector_cadd
--   c_adds = T.c_THLongVector_adds
--   c_cmul = T.c_THLongVector_cmul
--   c_muls = T.c_THLongVector_muls
--   c_cdiv = T.c_THLongVector_cdiv
--   c_divs = T.c_THLongVector_divs
--   c_copy = T.c_THLongVector_copy
--   c_neg = T.c_THLongVector_neg
--   c_abs = T.c_THLongVector_abs
--   c_log = T.c_THLongVector_log
--   c_lgamma = T.c_THLongVector_lgamma
--   c_log1p = T.c_THLongVector_log1p
--   c_sigmoid = T.c_THLongVector_sigmoid
--   c_exp = T.c_THLongVector_exp
--   c_erf = T.c_THLongVector_erf
--   c_erfinv = T.c_THLongVector_erfinv
--   c_cos = T.c_THLongVector_cos
--   c_acos = T.c_THLongVector_acos
--   c_cosh = T.c_THLongVector_cosh
--   c_sin = T.c_THLongVector_sin
--   c_asin = T.c_THLongVector_asin
--   c_sinh = T.c_THLongVector_sinh
--   c_tan = T.c_THLongVector_tan
--   c_atan = T.c_THLongVector_atan
--   c_tanh = T.c_THLongVector_tanh
--   c_pow = T.c_THLongVector_pow
--   c_sqrt = T.c_THLongVector_sqrt
--   c_rsqrt = T.c_THLongVector_rsqrt
--   c_ceil = T.c_THLongVector_ceil
--   c_floor = T.c_THLongVector_floor
--   c_round = T.c_THLongVector_round
--   c_trunc = T.c_THLongVector_trunc
--   c_frac = T.c_THLongVector_frac
--   c_cinv = T.c_THLongVector_cinv
--   -- c_vec = T.THLongVector_--
--   p_fill = T.p_THLongVector_fill
--   p_cadd = T.p_THLongVector_cadd
--   p_adds = T.p_THLongVector_adds
--   p_cmul = T.p_THLongVector_cmul
--   p_muls = T.p_THLongVector_muls
--   p_cdiv = T.p_THLongVector_cdiv
--   p_divs = T.p_THLongVector_divs
--   p_copy = T.p_THLongVector_copy
--   p_neg = T.p_THLongVector_neg
--   p_abs = T.p_THLongVector_abs
--   p_log = T.p_THLongVector_log
--   p_lgamma = T.p_THLongVector_lgamma
--   p_log1p = T.p_THLongVector_log1p
--   p_sigmoid = T.p_THLongVector_sigmoid
--   p_exp = T.p_THLongVector_exp
--   p_erf = T.p_THLongVector_erf
--   p_erfinv = T.p_THLongVector_erfinv
--   p_cos = T.p_THLongVector_cos
--   p_acos = T.p_THLongVector_acos
--   p_cosh = T.p_THLongVector_cosh
--   p_sin = T.p_THLongVector_sin
--   p_asin = T.p_THLongVector_asin
--   p_sinh = T.p_THLongVector_sinh
--   p_tan = T.p_THLongVector_tan
--   p_atan = T.p_THLongVector_atan
--   p_tanh = T.p_THLongVector_tanh
--   p_pow = T.p_THLongVector_pow
--   p_sqrt = T.p_THLongVector_sqrt
--   p_rsqrt = T.p_THLongVector_rsqrt
--   p_ceil = T.p_THLongVector_ceil
--   p_floor = T.p_THLongVector_floor
--   p_round = T.p_THLongVector_round
--   p_trunc = T.p_THLongVector_trunc
--   p_frac = T.p_THLongVector_frac
--   p_cinv = T.p_THLongVector_cinv
--   p_cinv = T.p_THLongVector_cinv

-- instance THVector CTHHalfVector where
--   c_fill = T.c_THHalfVector_fill
--   c_cadd = T.c_THHalfVector_cadd
--   c_adds = T.c_THHalfVector_adds
--   c_cmul = T.c_THHalfVector_cmul
--   c_muls = T.c_THHalfVector_muls
--   c_cdiv = T.c_THHalfVector_cdiv
--   c_divs = T.c_THHalfVector_divs
--   c_copy = T.c_THHalfVector_copy
--   c_neg = T.c_THHalfVector_neg
--   c_abs = T.c_THHalfVector_abs
--   c_log = T.c_THHalfVector_log
--   c_lgamma = T.c_THHalfVector_lgamma
--   c_log1p = T.c_THHalfVector_log1p
--   c_sigmoid = T.c_THHalfVector_sigmoid
--   c_exp = T.c_THHalfVector_exp
--   c_erf = T.c_THHalfVector_erf
--   c_erfinv = T.c_THHalfVector_erfinv
--   c_cos = T.c_THHalfVector_cos
--   c_acos = T.c_THHalfVector_acos
--   c_cosh = T.c_THHalfVector_cosh
--   c_sin = T.c_THHalfVector_sin
--   c_asin = T.c_THHalfVector_asin
--   c_sinh = T.c_THHalfVector_sinh
--   c_tan = T.c_THHalfVector_tan
--   c_atan = T.c_THHalfVector_atan
--   c_tanh = T.c_THHalfVector_tanh
--   c_pow = T.c_THHalfVector_pow
--   c_sqrt = T.c_THHalfVector_sqrt
--   c_rsqrt = T.c_THHalfVector_rsqrt
--   c_ceil = T.c_THHalfVector_ceil
--   c_floor = T.c_THHalfVector_floor
--   c_round = T.c_THHalfVector_round
--   c_trunc = T.c_THHalfVector_trunc
--   c_frac = T.c_THHalfVector_frac
--   c_cinv = T.c_THHalfVector_cinv
--   -- c_vec = T.THHalfVector_--
--   p_fill = T.p_THHalfVector_fill
--   p_cadd = T.p_THHalfVector_cadd
--   p_adds = T.p_THHalfVector_adds
--   p_cmul = T.p_THHalfVector_cmul
--   p_muls = T.p_THHalfVector_muls
--   p_cdiv = T.p_THHalfVector_cdiv
--   p_divs = T.p_THHalfVector_divs
--   p_copy = T.p_THHalfVector_copy
--   p_neg = T.p_THHalfVector_neg
--   p_abs = T.p_THHalfVector_abs
--   p_log = T.p_THHalfVector_log
--   p_lgamma = T.p_THHalfVector_lgamma
--   p_log1p = T.p_THHalfVector_log1p
--   p_sigmoid = T.p_THHalfVector_sigmoid
--   p_exp = T.p_THHalfVector_exp
--   p_erf = T.p_THHalfVector_erf
--   p_erfinv = T.p_THHalfVector_erfinv
--   p_cos = T.p_THHalfVector_cos
--   p_acos = T.p_THHalfVector_acos
--   p_cosh = T.p_THHalfVector_cosh
--   p_sin = T.p_THHalfVector_sin
--   p_asin = T.p_THHalfVector_asin
--   p_sinh = T.p_THHalfVector_sinh
--   p_tan = T.p_THHalfVector_tan
--   p_atan = T.p_THHalfVector_atan
--   p_tanh = T.p_THHalfVector_tanh
--   p_pow = T.p_THHalfVector_pow
--   p_sqrt = T.p_THHalfVector_sqrt
--   p_rsqrt = T.p_THHalfVector_rsqrt
--   p_ceil = T.p_THHalfVector_ceil
--   p_floor = T.p_THHalfVector_floor
--   p_round = T.p_THHalfVector_round
--   p_trunc = T.p_THHalfVector_trunc
--   p_frac = T.p_THHalfVector_frac
--   p_cinv = T.p_THHalfVector_cinv
--   p_cinv = T.p_THHalfVector_cinv
