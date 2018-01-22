{-# LANGUAGE ForeignFunctionInterface #-}

module THByteVector (
    c_THByteVector_fill,
    c_THByteVector_cadd,
    c_THByteVector_adds,
    c_THByteVector_cmul,
    c_THByteVector_muls,
    c_THByteVector_cdiv,
    c_THByteVector_divs,
    c_THByteVector_copy,
    c_THByteVector_normal_fill,
    c_THByteVector_digamma,
    c_THByteVector_trigamma,
    c_THByteVector_expm1,
    c_THByteVector_vectorDispatchInit,
    p_THByteVector_fill,
    p_THByteVector_cadd,
    p_THByteVector_adds,
    p_THByteVector_cmul,
    p_THByteVector_muls,
    p_THByteVector_cdiv,
    p_THByteVector_divs,
    p_THByteVector_copy,
    p_THByteVector_normal_fill,
    p_THByteVector_digamma,
    p_THByteVector_trigamma,
    p_THByteVector_expm1,
    p_THByteVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THByteVector_fill : x c n -> void
foreign import ccall "THVector.h THByteVector_fill"
  c_THByteVector_fill :: Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THByteVector_cadd"
  c_THByteVector_cadd :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_adds : y x c n -> void
foreign import ccall "THVector.h THByteVector_adds"
  c_THByteVector_adds :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_cmul : z x y n -> void
foreign import ccall "THVector.h THByteVector_cmul"
  c_THByteVector_cmul :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_muls : y x c n -> void
foreign import ccall "THVector.h THByteVector_muls"
  c_THByteVector_muls :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THByteVector_cdiv"
  c_THByteVector_cdiv :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_divs : y x c n -> void
foreign import ccall "THVector.h THByteVector_divs"
  c_THByteVector_divs :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_copy : y x n -> void
foreign import ccall "THVector.h THByteVector_copy"
  c_THByteVector_copy :: Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_normal_fill : data size generator mean stddev -> void
foreign import ccall "THVector.h THByteVector_normal_fill"
  c_THByteVector_normal_fill :: Ptr CChar -> CLLong -> Ptr CTHGenerator -> CChar -> CChar -> IO ()

-- |c_THByteVector_digamma : y x n -> void
foreign import ccall "THVector.h THByteVector_digamma"
  c_THByteVector_digamma :: Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_trigamma : y x n -> void
foreign import ccall "THVector.h THByteVector_trigamma"
  c_THByteVector_trigamma :: Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_expm1 : y x n -> void
foreign import ccall "THVector.h THByteVector_expm1"
  c_THByteVector_expm1 :: Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THByteVector_vectorDispatchInit"
  c_THByteVector_vectorDispatchInit :: IO ()

-- |p_THByteVector_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THByteVector_fill"
  p_THByteVector_fill :: FunPtr (Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THByteVector_cadd"
  p_THByteVector_cadd :: FunPtr (Ptr CChar -> Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THByteVector_adds"
  p_THByteVector_adds :: FunPtr (Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THByteVector_cmul"
  p_THByteVector_cmul :: FunPtr (Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THByteVector_muls"
  p_THByteVector_muls :: FunPtr (Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THByteVector_cdiv"
  p_THByteVector_cdiv :: FunPtr (Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THByteVector_divs"
  p_THByteVector_divs :: FunPtr (Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THByteVector_copy"
  p_THByteVector_copy :: FunPtr (Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THByteVector_normal_fill"
  p_THByteVector_normal_fill :: FunPtr (Ptr CChar -> CLLong -> Ptr CTHGenerator -> CChar -> CChar -> IO ())

-- |p_THByteVector_digamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THByteVector_digamma"
  p_THByteVector_digamma :: FunPtr (Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_trigamma : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THByteVector_trigamma"
  p_THByteVector_trigamma :: FunPtr (Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_expm1 : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THByteVector_expm1"
  p_THByteVector_expm1 :: FunPtr (Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- |p_THByteVector_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THByteVector_vectorDispatchInit"
  p_THByteVector_vectorDispatchInit :: FunPtr (IO ())