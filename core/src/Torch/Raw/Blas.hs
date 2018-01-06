module Torch.Raw.Blas
  ( THBlas(..)
  , module X
  ) where

import Torch.Raw.Internal as X

-- t == CDouble
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
