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
    c_THDoubleVector_vectorDispatchInit,
    p_THDoubleVector_fill,
    p_THDoubleVector_cadd,
    p_THDoubleVector_adds,
    p_THDoubleVector_cmul,
    p_THDoubleVector_muls,
    p_THDoubleVector_cdiv,
    p_THDoubleVector_divs,
    p_THDoubleVector_copy,
    p_THDoubleVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleVector_fill : x c n -> void
foreign import ccall unsafe "THVector.h THDoubleVector_fill"
  c_THDoubleVector_fill :: Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cadd : z x y c n -> void
foreign import ccall unsafe "THVector.h THDoubleVector_cadd"
  c_THDoubleVector_cadd :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_adds : y x c n -> void
foreign import ccall unsafe "THVector.h THDoubleVector_adds"
  c_THDoubleVector_adds :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cmul : z x y n -> void
foreign import ccall unsafe "THVector.h THDoubleVector_cmul"
  c_THDoubleVector_cmul :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_muls : y x c n -> void
foreign import ccall unsafe "THVector.h THDoubleVector_muls"
  c_THDoubleVector_muls :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_cdiv : z x y n -> void
foreign import ccall unsafe "THVector.h THDoubleVector_cdiv"
  c_THDoubleVector_cdiv :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_divs : y x c n -> void
foreign import ccall unsafe "THVector.h THDoubleVector_divs"
  c_THDoubleVector_divs :: Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_copy : y x n -> void
foreign import ccall unsafe "THVector.h THDoubleVector_copy"
  c_THDoubleVector_copy :: Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ()

-- |c_THDoubleVector_vectorDispatchInit :  -> void
foreign import ccall unsafe "THVector.h THDoubleVector_vectorDispatchInit"
  c_THDoubleVector_vectorDispatchInit :: IO ()

-- |p_THDoubleVector_fill : Pointer to function x c n -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_fill"
  p_THDoubleVector_fill :: FunPtr (Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cadd : Pointer to function z x y c n -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_cadd"
  p_THDoubleVector_cadd :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_adds : Pointer to function y x c n -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_adds"
  p_THDoubleVector_adds :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cmul : Pointer to function z x y n -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_cmul"
  p_THDoubleVector_cmul :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_muls : Pointer to function y x c n -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_muls"
  p_THDoubleVector_muls :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_cdiv : Pointer to function z x y n -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_cdiv"
  p_THDoubleVector_cdiv :: FunPtr (Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_divs : Pointer to function y x c n -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_divs"
  p_THDoubleVector_divs :: FunPtr (Ptr CDouble -> Ptr CDouble -> CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_copy : Pointer to function y x n -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_copy"
  p_THDoubleVector_copy :: FunPtr (Ptr CDouble -> Ptr CDouble -> CPtrdiff -> IO ())

-- |p_THDoubleVector_vectorDispatchInit : Pointer to function  -> void
foreign import ccall unsafe "THVector.h &THDoubleVector_vectorDispatchInit"
  p_THDoubleVector_vectorDispatchInit :: FunPtr (IO ())