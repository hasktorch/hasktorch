{-# LANGUAGE TypeSynonymInstances #-}

module Torch.Raw.Blas
  ( THBlas(..)
  , module X
  ) where

import Torch.Raw.Internal as X

import qualified THLongBlas as T
import qualified THByteBlas as T
import qualified THIntBlas as T
import qualified THShortBlas as T
import qualified THHalfBlas as T
import qualified THByteBlas as T
import qualified THDoubleBlas as T
import qualified THFloatBlas as T


-- t == T.CDouble
class THBlas t where
  c_swap :: CLLong -> Ptr t -> CLLong -> Ptr t -> CLLong -> IO ()
  c_scal :: CLLong -> t -> Ptr t -> CLLong -> IO ()
  c_copy :: CLLong -> Ptr t -> CLLong -> Ptr t -> CLLong -> IO ()
  c_axpy :: CLLong -> t -> Ptr t -> CLLong -> Ptr t -> CLLong -> IO ()
  c_dot  :: CLLong -> Ptr t -> CLLong -> Ptr t -> CLLong -> t
  c_gemv :: CChar -> CLLong -> CLLong -> t -> Ptr t -> CLLong -> Ptr t -> CLLong -> t -> Ptr t -> CLLong -> IO ()
  c_ger  :: CLLong -> CLLong -> t -> Ptr t -> CLLong -> Ptr t -> CLLong -> Ptr t -> CLLong -> IO ()

  p_swap :: FunPtr (CLLong -> Ptr t -> CLLong -> Ptr t -> CLLong -> IO ())
  p_scal :: FunPtr (CLLong -> t -> Ptr t -> CLLong -> IO ())
  p_copy :: FunPtr (CLLong -> Ptr t -> CLLong -> Ptr t -> CLLong -> IO ())
  p_axpy :: FunPtr (CLLong -> t -> Ptr t -> CLLong -> Ptr t -> CLLong -> IO ())
  p_dot  :: FunPtr (CLLong -> Ptr t -> CLLong -> Ptr t -> CLLong -> t)
  p_gemv :: FunPtr (CChar -> CLLong -> CLLong -> t -> Ptr t -> CLLong -> Ptr t -> CLLong -> t -> Ptr t -> CLLong -> IO ())
  p_ger  :: FunPtr (CLLong -> CLLong -> t -> Ptr t -> CLLong -> Ptr t -> CLLong -> Ptr t -> CLLong -> IO ())
  p_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> t -> Ptr t -> CLLong -> Ptr t -> CLLong -> t -> Ptr t -> CLLong -> IO ())

instance THBlas THLongBlas where
  c_swap = T.c_THLongBlas_swap
  c_scal = T.c_THLongBlas_scal
  c_copy = T.c_THLongBlas_copy
  c_axpy = T.c_THLongBlas_axpy
  c_dot = T.c_THLongBlas_dot
  c_gemv = T.c_THLongBlas_gemv
  c_ger = T.c_THLongBlas_ger
  p_swap = T.p_THLongBlas_swap
  p_scal = T.p_THLongBlas_scal
  p_copy = T.p_THLongBlas_copy
  p_axpy = T.p_THLongBlas_axpy
  p_dot = T.p_THLongBlas_dot
  p_gemv = T.p_THLongBlas_gemv
  p_ger = T.p_THLongBlas_ger
  p_gemm = T.p_THLongBlas_gemm

instance THBlas THShortBlas where
  c_swap = T.c_THShortBlas_swap
  c_scal = T.c_THShortBlas_scal
  c_copy = T.c_THShortBlas_copy
  c_axpy = T.c_THShortBlas_axpy
  c_dot = T.c_THShortBlas_dot
  c_gemv = T.c_THShortBlas_gemv
  c_ger = T.c_THShortBlas_ger
  p_swap = T.p_THShortBlas_swap
  p_scal = T.p_THShortBlas_scal
  p_copy = T.p_THShortBlas_copy
  p_axpy = T.p_THShortBlas_axpy
  p_dot = T.p_THShortBlas_dot
  p_gemv = T.p_THShortBlas_gemv
  p_ger = T.p_THShortBlas_ger
  p_gemm = T.p_THShortBlas_gemm

instance THBlas THIntBlas where
  c_swap = T.c_THIntBlas_swap
  c_scal = T.c_THIntBlas_scal
  c_copy = T.c_THIntBlas_copy
  c_axpy = T.c_THIntBlas_axpy
  c_dot = T.c_THIntBlas_dot
  c_gemv = T.c_THIntBlas_gemv
  c_ger = T.c_THIntBlas_ger
  p_swap = T.p_THIntBlas_swap
  p_scal = T.p_THIntBlas_scal
  p_copy = T.p_THIntBlas_copy
  p_axpy = T.p_THIntBlas_axpy
  p_dot = T.p_THIntBlas_dot
  p_gemv = T.p_THIntBlas_gemv
  p_ger = T.p_THIntBlas_ger
  p_gemm = T.p_THIntBlas_gemm

instance THBlas THByteBlas where
  c_swap = T.c_THByteBlas_swap
  c_scal = T.c_THByteBlas_scal
  c_copy = T.c_THByteBlas_copy
  c_axpy = T.c_THByteBlas_axpy
  c_dot = T.c_THByteBlas_dot
  c_gemv = T.c_THByteBlas_gemv
  c_ger = T.c_THByteBlas_ger
  p_swap = T.p_THByteBlas_swap
  p_scal = T.p_THByteBlas_scal
  p_copy = T.p_THByteBlas_copy
  p_axpy = T.p_THByteBlas_axpy
  p_dot = T.p_THByteBlas_dot
  p_gemv = T.p_THByteBlas_gemv
  p_ger = T.p_THByteBlas_ger
  p_gemm = T.p_THByteBlas_gemm

-- Half and Short are synonyms for CShort
-- instance THBlas THHalfBlas where
--   c_swap = T.c_THHalfBlas_swap
--   c_scal = T.c_THHalfBlas_scal
--   c_copy = T.c_THHalfBlas_copy
--   c_axpy = T.c_THHalfBlas_axpy
--   c_dot = T.c_THHalfBlas_dot
--   c_gemv = T.c_THHalfBlas_gemv
--   c_ger = T.c_THHalfBlas_ger
--   p_swap = T.p_THHalfBlas_swap
--   p_scal = T.p_THHalfBlas_scal
--   p_copy = T.p_THHalfBlas_copy
--   p_axpy = T.p_THHalfBlas_axpy
--   p_dot = T.p_THHalfBlas_dot
--   p_gemv = T.p_THHalfBlas_gemv
--   p_ger = T.p_THHalfBlas_ger
--   p_gemm = T.p_THHalfBlas_gemm

instance THBlas THFloatBlas where
  c_swap = T.c_THFloatBlas_swap
  c_scal = T.c_THFloatBlas_scal
  c_copy = T.c_THFloatBlas_copy
  c_axpy = T.c_THFloatBlas_axpy
  c_dot = T.c_THFloatBlas_dot
  c_gemv = T.c_THFloatBlas_gemv
  c_ger = T.c_THFloatBlas_ger
  p_swap = T.p_THFloatBlas_swap
  p_scal = T.p_THFloatBlas_scal
  p_copy = T.p_THFloatBlas_copy
  p_axpy = T.p_THFloatBlas_axpy
  p_dot = T.p_THFloatBlas_dot
  p_gemv = T.p_THFloatBlas_gemv
  p_ger = T.p_THFloatBlas_ger
  p_gemm = T.p_THFloatBlas_gemm

instance THBlas THDoubleBlas where
  c_swap = T.c_THDoubleBlas_swap
  c_scal = T.c_THDoubleBlas_scal
  c_copy = T.c_THDoubleBlas_copy
  c_axpy = T.c_THDoubleBlas_axpy
  c_dot = T.c_THDoubleBlas_dot
  c_gemv = T.c_THDoubleBlas_gemv
  c_ger = T.c_THDoubleBlas_ger
  p_swap = T.p_THDoubleBlas_swap
  p_scal = T.p_THDoubleBlas_scal
  p_copy = T.p_THDoubleBlas_copy
  p_axpy = T.p_THDoubleBlas_axpy
  p_dot = T.p_THDoubleBlas_dot
  p_gemv = T.p_THDoubleBlas_gemv
  p_ger = T.p_THDoubleBlas_ger
  p_gemm = T.p_THDoubleBlas_gemm
