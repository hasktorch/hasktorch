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
    c_THHalfVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfVector_fill : x c n -> void
foreign import ccall "THVector.h THHalfVector_fill"
  c_THHalfVector_fill :: Ptr THHalf -> THHalf -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THHalfVector_cadd"
  c_THHalfVector_cadd :: Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> THHalf -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfVector_adds : y x c n -> void
foreign import ccall "THVector.h THHalfVector_adds"
  c_THHalfVector_adds :: Ptr THHalf -> Ptr THHalf -> THHalf -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfVector_cmul : z x y n -> void
foreign import ccall "THVector.h THHalfVector_cmul"
  c_THHalfVector_cmul :: Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfVector_muls : y x c n -> void
foreign import ccall "THVector.h THHalfVector_muls"
  c_THHalfVector_muls :: Ptr THHalf -> Ptr THHalf -> THHalf -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THHalfVector_cdiv"
  c_THHalfVector_cdiv :: Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfVector_divs : y x c n -> void
foreign import ccall "THVector.h THHalfVector_divs"
  c_THHalfVector_divs :: Ptr THHalf -> Ptr THHalf -> THHalf -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfVector_copy : y x n -> void
foreign import ccall "THVector.h THHalfVector_copy"
  c_THHalfVector_copy :: Ptr THHalf -> Ptr THHalf -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THHalfVector_vectorDispatchInit"
  c_THHalfVector_vectorDispatchInit :: IO ()