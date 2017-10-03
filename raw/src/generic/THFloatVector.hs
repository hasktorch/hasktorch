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
    c_THFloatVector_vectorDispatchInit,
    p_THFloatVector_fill,
    p_THFloatVector_cadd,
    p_THFloatVector_adds,
    p_THFloatVector_cmul,
    p_THFloatVector_muls,
    p_THFloatVector_cdiv,
    p_THFloatVector_divs,
    p_THFloatVector_copy,
    p_THFloatVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatVector_fill : x c n -> void
foreign import ccall unsafe "THVector.h THFloatVector_fill"
  c_THFloatVector_fill :: Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cadd : z x y c n -> void
foreign import ccall unsafe "THVector.h THFloatVector_cadd"
  c_THFloatVector_cadd :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_adds : y x c n -> void
foreign import ccall unsafe "THVector.h THFloatVector_adds"
  c_THFloatVector_adds :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cmul : z x y n -> void
foreign import ccall unsafe "THVector.h THFloatVector_cmul"
  c_THFloatVector_cmul :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_muls : y x c n -> void
foreign import ccall unsafe "THVector.h THFloatVector_muls"
  c_THFloatVector_muls :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_cdiv : z x y n -> void
foreign import ccall unsafe "THVector.h THFloatVector_cdiv"
  c_THFloatVector_cdiv :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_divs : y x c n -> void
foreign import ccall unsafe "THVector.h THFloatVector_divs"
  c_THFloatVector_divs :: Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_copy : y x n -> void
foreign import ccall unsafe "THVector.h THFloatVector_copy"
  c_THFloatVector_copy :: Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ()

-- |c_THFloatVector_vectorDispatchInit :  -> void
foreign import ccall unsafe "THVector.h THFloatVector_vectorDispatchInit"
  c_THFloatVector_vectorDispatchInit :: IO ()

-- |p_THFloatVector_fill : Pointer to function x c n -> void
foreign import ccall unsafe "THVector.h &THFloatVector_fill"
  p_THFloatVector_fill :: FunPtr (Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cadd : Pointer to function z x y c n -> void
foreign import ccall unsafe "THVector.h &THFloatVector_cadd"
  p_THFloatVector_cadd :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_adds : Pointer to function y x c n -> void
foreign import ccall unsafe "THVector.h &THFloatVector_adds"
  p_THFloatVector_adds :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cmul : Pointer to function z x y n -> void
foreign import ccall unsafe "THVector.h &THFloatVector_cmul"
  p_THFloatVector_cmul :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_muls : Pointer to function y x c n -> void
foreign import ccall unsafe "THVector.h &THFloatVector_muls"
  p_THFloatVector_muls :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_cdiv : Pointer to function z x y n -> void
foreign import ccall unsafe "THVector.h &THFloatVector_cdiv"
  p_THFloatVector_cdiv :: FunPtr (Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_divs : Pointer to function y x c n -> void
foreign import ccall unsafe "THVector.h &THFloatVector_divs"
  p_THFloatVector_divs :: FunPtr (Ptr CFloat -> Ptr CFloat -> CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_copy : Pointer to function y x n -> void
foreign import ccall unsafe "THVector.h &THFloatVector_copy"
  p_THFloatVector_copy :: FunPtr (Ptr CFloat -> Ptr CFloat -> CPtrdiff -> IO ())

-- |p_THFloatVector_vectorDispatchInit : Pointer to function  -> void
foreign import ccall unsafe "THVector.h &THFloatVector_vectorDispatchInit"
  p_THFloatVector_vectorDispatchInit :: FunPtr (IO ())