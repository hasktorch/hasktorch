{-# LANGUAGE ForeignFunctionInterface #-}

module THLongVector (
    c_THLongVector_fill,
    c_THLongVector_cadd,
    c_THLongVector_adds,
    c_THLongVector_cmul,
    c_THLongVector_muls,
    c_THLongVector_cdiv,
    c_THLongVector_divs,
    c_THLongVector_copy,
    c_THLongVector_vectorDispatchInit,
    p_THLongVector_fill,
    p_THLongVector_cadd,
    p_THLongVector_adds,
    p_THLongVector_cmul,
    p_THLongVector_muls,
    p_THLongVector_cdiv,
    p_THLongVector_divs,
    p_THLongVector_copy,
    p_THLongVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongVector_fill : x c n -> void
foreign import ccall unsafe "THVector.h THLongVector_fill"
  c_THLongVector_fill :: Ptr CLong -> CLong -> CPtrdiff -> IO ()

-- |c_THLongVector_cadd : z x y c n -> void
foreign import ccall unsafe "THVector.h THLongVector_cadd"
  c_THLongVector_cadd :: Ptr CLong -> Ptr CLong -> Ptr CLong -> CLong -> CPtrdiff -> IO ()

-- |c_THLongVector_adds : y x c n -> void
foreign import ccall unsafe "THVector.h THLongVector_adds"
  c_THLongVector_adds :: Ptr CLong -> Ptr CLong -> CLong -> CPtrdiff -> IO ()

-- |c_THLongVector_cmul : z x y n -> void
foreign import ccall unsafe "THVector.h THLongVector_cmul"
  c_THLongVector_cmul :: Ptr CLong -> Ptr CLong -> Ptr CLong -> CPtrdiff -> IO ()

-- |c_THLongVector_muls : y x c n -> void
foreign import ccall unsafe "THVector.h THLongVector_muls"
  c_THLongVector_muls :: Ptr CLong -> Ptr CLong -> CLong -> CPtrdiff -> IO ()

-- |c_THLongVector_cdiv : z x y n -> void
foreign import ccall unsafe "THVector.h THLongVector_cdiv"
  c_THLongVector_cdiv :: Ptr CLong -> Ptr CLong -> Ptr CLong -> CPtrdiff -> IO ()

-- |c_THLongVector_divs : y x c n -> void
foreign import ccall unsafe "THVector.h THLongVector_divs"
  c_THLongVector_divs :: Ptr CLong -> Ptr CLong -> CLong -> CPtrdiff -> IO ()

-- |c_THLongVector_copy : y x n -> void
foreign import ccall unsafe "THVector.h THLongVector_copy"
  c_THLongVector_copy :: Ptr CLong -> Ptr CLong -> CPtrdiff -> IO ()

-- |c_THLongVector_vectorDispatchInit :  -> void
foreign import ccall unsafe "THVector.h THLongVector_vectorDispatchInit"
  c_THLongVector_vectorDispatchInit :: IO ()

-- |p_THLongVector_fill : Pointer to x c n -> void
foreign import ccall unsafe "THVector.h &THLongVector_fill"
  p_THLongVector_fill :: FunPtr (Ptr CLong -> CLong -> CPtrdiff -> IO ())

-- |p_THLongVector_cadd : Pointer to z x y c n -> void
foreign import ccall unsafe "THVector.h &THLongVector_cadd"
  p_THLongVector_cadd :: FunPtr (Ptr CLong -> Ptr CLong -> Ptr CLong -> CLong -> CPtrdiff -> IO ())

-- |p_THLongVector_adds : Pointer to y x c n -> void
foreign import ccall unsafe "THVector.h &THLongVector_adds"
  p_THLongVector_adds :: FunPtr (Ptr CLong -> Ptr CLong -> CLong -> CPtrdiff -> IO ())

-- |p_THLongVector_cmul : Pointer to z x y n -> void
foreign import ccall unsafe "THVector.h &THLongVector_cmul"
  p_THLongVector_cmul :: FunPtr (Ptr CLong -> Ptr CLong -> Ptr CLong -> CPtrdiff -> IO ())

-- |p_THLongVector_muls : Pointer to y x c n -> void
foreign import ccall unsafe "THVector.h &THLongVector_muls"
  p_THLongVector_muls :: FunPtr (Ptr CLong -> Ptr CLong -> CLong -> CPtrdiff -> IO ())

-- |p_THLongVector_cdiv : Pointer to z x y n -> void
foreign import ccall unsafe "THVector.h &THLongVector_cdiv"
  p_THLongVector_cdiv :: FunPtr (Ptr CLong -> Ptr CLong -> Ptr CLong -> CPtrdiff -> IO ())

-- |p_THLongVector_divs : Pointer to y x c n -> void
foreign import ccall unsafe "THVector.h &THLongVector_divs"
  p_THLongVector_divs :: FunPtr (Ptr CLong -> Ptr CLong -> CLong -> CPtrdiff -> IO ())

-- |p_THLongVector_copy : Pointer to y x n -> void
foreign import ccall unsafe "THVector.h &THLongVector_copy"
  p_THLongVector_copy :: FunPtr (Ptr CLong -> Ptr CLong -> CPtrdiff -> IO ())

-- |p_THLongVector_vectorDispatchInit : Pointer to  -> void
foreign import ccall unsafe "THVector.h &THLongVector_vectorDispatchInit"
  p_THLongVector_vectorDispatchInit :: FunPtr (IO ())