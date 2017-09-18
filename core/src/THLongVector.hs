{-# LANGUAGE ForeignFunctionInterface#-}

module THLongVector (
    c_THLongVector_fill,
    c_THLongVector_cadd,
    c_THLongVector_adds,
    c_THLongVector_cmul,
    c_THLongVector_muls,
    c_THLongVector_cdiv,
    c_THLongVector_divs,
    c_THLongVector_copy,
    c_THLongVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongVector_fill : x c n -> void
foreign import ccall "THVector.h THLongVector_fill"
  c_THLongVector_fill :: Ptr CLong -> CLong -> Ptr CTHLongStorage -> IO ()

-- |c_THLongVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THLongVector_cadd"
  c_THLongVector_cadd :: Ptr CLong -> Ptr CLong -> Ptr CLong -> CLong -> Ptr CTHLongStorage -> IO ()

-- |c_THLongVector_adds : y x c n -> void
foreign import ccall "THVector.h THLongVector_adds"
  c_THLongVector_adds :: Ptr CLong -> Ptr CLong -> CLong -> Ptr CTHLongStorage -> IO ()

-- |c_THLongVector_cmul : z x y n -> void
foreign import ccall "THVector.h THLongVector_cmul"
  c_THLongVector_cmul :: Ptr CLong -> Ptr CLong -> Ptr CLong -> Ptr CTHLongStorage -> IO ()

-- |c_THLongVector_muls : y x c n -> void
foreign import ccall "THVector.h THLongVector_muls"
  c_THLongVector_muls :: Ptr CLong -> Ptr CLong -> CLong -> Ptr CTHLongStorage -> IO ()

-- |c_THLongVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THLongVector_cdiv"
  c_THLongVector_cdiv :: Ptr CLong -> Ptr CLong -> Ptr CLong -> Ptr CTHLongStorage -> IO ()

-- |c_THLongVector_divs : y x c n -> void
foreign import ccall "THVector.h THLongVector_divs"
  c_THLongVector_divs :: Ptr CLong -> Ptr CLong -> CLong -> Ptr CTHLongStorage -> IO ()

-- |c_THLongVector_copy : y x n -> void
foreign import ccall "THVector.h THLongVector_copy"
  c_THLongVector_copy :: Ptr CLong -> Ptr CLong -> Ptr CTHLongStorage -> IO ()

-- |c_THLongVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THLongVector_vectorDispatchInit"
  c_THLongVector_vectorDispatchInit :: IO ()