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
    c_THFloatVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatVector_fill : x c n -> void
foreign import ccall "THVector.h THFloatVector_fill"
  c_THFloatVector_fill :: Ptr CFloat -> CFloat -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THFloatVector_cadd"
  c_THFloatVector_cadd :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CFloat -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatVector_adds : y x c n -> void
foreign import ccall "THVector.h THFloatVector_adds"
  c_THFloatVector_adds :: Ptr CFloat -> Ptr CFloat -> CFloat -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatVector_cmul : z x y n -> void
foreign import ccall "THVector.h THFloatVector_cmul"
  c_THFloatVector_cmul :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatVector_muls : y x c n -> void
foreign import ccall "THVector.h THFloatVector_muls"
  c_THFloatVector_muls :: Ptr CFloat -> Ptr CFloat -> CFloat -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THFloatVector_cdiv"
  c_THFloatVector_cdiv :: Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatVector_divs : y x c n -> void
foreign import ccall "THVector.h THFloatVector_divs"
  c_THFloatVector_divs :: Ptr CFloat -> Ptr CFloat -> CFloat -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatVector_copy : y x n -> void
foreign import ccall "THVector.h THFloatVector_copy"
  c_THFloatVector_copy :: Ptr CFloat -> Ptr CFloat -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THFloatVector_vectorDispatchInit"
  c_THFloatVector_vectorDispatchInit :: IO ()