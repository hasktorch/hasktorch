{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfVector (
    c_THHalfVector_fill,
    c_THHalfVector_cadd,
    c_THHalfVector_adds,
    c_THHalfVector_cmul,
    c_THHalfVector_muls,
    c_THHalfVector_cdiv,
    c_THHalfVector_divs,
    c_THHalfVector_copy,
    c_THHalfVector_erf,
    c_THHalfVector_erfinv,
    c_THHalfVector_vectorDispatchInit,
    p_THHalfVector_fill,
    p_THHalfVector_cadd,
    p_THHalfVector_adds,
    p_THHalfVector_cmul,
    p_THHalfVector_muls,
    p_THHalfVector_cdiv,
    p_THHalfVector_divs,
    p_THHalfVector_copy,
    p_THHalfVector_erf,
    p_THHalfVector_erfinv,
    p_THHalfVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THHalfVector_fill : x c n -> void
foreign import ccall "THVector.h THHalfVector_fill"
  c_THHalfVector_fill :: Ptr THHalf -> THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THHalfVector_cadd"
  c_THHalfVector_cadd :: Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_adds : y x c n -> void
foreign import ccall "THVector.h THHalfVector_adds"
  c_THHalfVector_adds :: Ptr THHalf -> Ptr THHalf -> THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_cmul : z x y n -> void
foreign import ccall "THVector.h THHalfVector_cmul"
  c_THHalfVector_cmul :: Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_muls : y x c n -> void
foreign import ccall "THVector.h THHalfVector_muls"
  c_THHalfVector_muls :: Ptr THHalf -> Ptr THHalf -> THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THHalfVector_cdiv"
  c_THHalfVector_cdiv :: Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_divs : y x c n -> void
foreign import ccall "THVector.h THHalfVector_divs"
  c_THHalfVector_divs :: Ptr THHalf -> Ptr THHalf -> THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_copy : y x n -> void
foreign import ccall "THVector.h THHalfVector_copy"
  c_THHalfVector_copy :: Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_erf : y x n -> void
foreign import ccall "THVector.h THHalfVector_erf"
  c_THHalfVector_erf :: Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_erfinv : y x n -> void
foreign import ccall "THVector.h THHalfVector_erfinv"
  c_THHalfVector_erfinv :: Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ()

-- |c_THHalfVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THHalfVector_vectorDispatchInit"
  c_THHalfVector_vectorDispatchInit :: IO ()

-- |p_THHalfVector_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THHalfVector_fill"
  p_THHalfVector_fill :: FunPtr (Ptr THHalf -> THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THHalfVector_cadd"
  p_THHalfVector_cadd :: FunPtr (Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THHalfVector_adds"
  p_THHalfVector_adds :: FunPtr (Ptr THHalf -> Ptr THHalf -> THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THHalfVector_cmul"
  p_THHalfVector_cmul :: FunPtr (Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THHalfVector_muls"
  p_THHalfVector_muls :: FunPtr (Ptr THHalf -> Ptr THHalf -> THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THHalfVector_cdiv"
  p_THHalfVector_cdiv :: FunPtr (Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THHalfVector_divs"
  p_THHalfVector_divs :: FunPtr (Ptr THHalf -> Ptr THHalf -> THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THHalfVector_copy"
  p_THHalfVector_copy :: FunPtr (Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_erf : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THHalfVector_erf"
  p_THHalfVector_erf :: FunPtr (Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_erfinv : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THHalfVector_erfinv"
  p_THHalfVector_erfinv :: FunPtr (Ptr THHalf -> Ptr THHalf -> CPtrdiff -> IO ())

-- |p_THHalfVector_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THHalfVector_vectorDispatchInit"
  p_THHalfVector_vectorDispatchInit :: FunPtr (IO ())