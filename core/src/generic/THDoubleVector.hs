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
    c_THDoubleVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleVector_fill : x c n -> void
foreign import ccall "THVector.h THDoubleVector_fill"
  c_THDoubleVector_fill :: Ptr CDouble -> CDouble -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THDoubleVector_cadd"
  c_THDoubleVector_cadd :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CDouble -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleVector_adds : y x c n -> void
foreign import ccall "THVector.h THDoubleVector_adds"
  c_THDoubleVector_adds :: Ptr CDouble -> Ptr CDouble -> CDouble -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleVector_cmul : z x y n -> void
foreign import ccall "THVector.h THDoubleVector_cmul"
  c_THDoubleVector_cmul :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleVector_muls : y x c n -> void
foreign import ccall "THVector.h THDoubleVector_muls"
  c_THDoubleVector_muls :: Ptr CDouble -> Ptr CDouble -> CDouble -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THDoubleVector_cdiv"
  c_THDoubleVector_cdiv :: Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleVector_divs : y x c n -> void
foreign import ccall "THVector.h THDoubleVector_divs"
  c_THDoubleVector_divs :: Ptr CDouble -> Ptr CDouble -> CDouble -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleVector_copy : y x n -> void
foreign import ccall "THVector.h THDoubleVector_copy"
  c_THDoubleVector_copy :: Ptr CDouble -> Ptr CDouble -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THDoubleVector_vectorDispatchInit"
  c_THDoubleVector_vectorDispatchInit :: IO ()