{-# LANGUAGE ForeignFunctionInterface #-}

module THIntVector (
    c_THIntVector_fill,
    c_THIntVector_cadd,
    c_THIntVector_adds,
    c_THIntVector_cmul,
    c_THIntVector_muls,
    c_THIntVector_cdiv,
    c_THIntVector_divs,
    c_THIntVector_copy,
    c_THIntVector_abs,
    c_THIntVector_neg,
    c_THIntVector_vectorDispatchInit,
    p_THIntVector_fill,
    p_THIntVector_cadd,
    p_THIntVector_adds,
    p_THIntVector_cmul,
    p_THIntVector_muls,
    p_THIntVector_cdiv,
    p_THIntVector_divs,
    p_THIntVector_copy,
    p_THIntVector_abs,
    p_THIntVector_neg,
    p_THIntVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THIntVector_fill : x c n -> void
foreign import ccall "THVector.h THIntVector_fill"
  c_THIntVector_fill :: Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THIntVector_cadd"
  c_THIntVector_cadd :: Ptr CInt -> Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_adds : y x c n -> void
foreign import ccall "THVector.h THIntVector_adds"
  c_THIntVector_adds :: Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_cmul : z x y n -> void
foreign import ccall "THVector.h THIntVector_cmul"
  c_THIntVector_cmul :: Ptr CInt -> Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_muls : y x c n -> void
foreign import ccall "THVector.h THIntVector_muls"
  c_THIntVector_muls :: Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THIntVector_cdiv"
  c_THIntVector_cdiv :: Ptr CInt -> Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_divs : y x c n -> void
foreign import ccall "THVector.h THIntVector_divs"
  c_THIntVector_divs :: Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_copy : y x n -> void
foreign import ccall "THVector.h THIntVector_copy"
  c_THIntVector_copy :: Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_abs : y x n -> void
foreign import ccall "THVector.h THIntVector_abs"
  c_THIntVector_abs :: Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_neg : y x n -> void
foreign import ccall "THVector.h THIntVector_neg"
  c_THIntVector_neg :: Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- |c_THIntVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THIntVector_vectorDispatchInit"
  c_THIntVector_vectorDispatchInit :: IO ()

-- |p_THIntVector_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THIntVector_fill"
  p_THIntVector_fill :: FunPtr (Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THIntVector_cadd"
  p_THIntVector_cadd :: FunPtr (Ptr CInt -> Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THIntVector_adds"
  p_THIntVector_adds :: FunPtr (Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THIntVector_cmul"
  p_THIntVector_cmul :: FunPtr (Ptr CInt -> Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THIntVector_muls"
  p_THIntVector_muls :: FunPtr (Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THIntVector_cdiv"
  p_THIntVector_cdiv :: FunPtr (Ptr CInt -> Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THIntVector_divs"
  p_THIntVector_divs :: FunPtr (Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THIntVector_copy"
  p_THIntVector_copy :: FunPtr (Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THIntVector_abs"
  p_THIntVector_abs :: FunPtr (Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THIntVector_neg"
  p_THIntVector_neg :: FunPtr (Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- |p_THIntVector_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THIntVector_vectorDispatchInit"
  p_THIntVector_vectorDispatchInit :: FunPtr (IO ())