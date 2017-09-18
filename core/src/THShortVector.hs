{-# LANGUAGE ForeignFunctionInterface#-}

module THShortVector (
    c_THShortVector_fill,
    c_THShortVector_cadd,
    c_THShortVector_adds,
    c_THShortVector_cmul,
    c_THShortVector_muls,
    c_THShortVector_cdiv,
    c_THShortVector_divs,
    c_THShortVector_copy,
    c_THShortVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortVector_fill : x c n -> void
foreign import ccall "THVector.h THShortVector_fill"
  c_THShortVector_fill :: Ptr CShort -> CShort -> Ptr CTHShortStorage -> IO ()

-- |c_THShortVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THShortVector_cadd"
  c_THShortVector_cadd :: Ptr CShort -> Ptr CShort -> Ptr CShort -> CShort -> Ptr CTHShortStorage -> IO ()

-- |c_THShortVector_adds : y x c n -> void
foreign import ccall "THVector.h THShortVector_adds"
  c_THShortVector_adds :: Ptr CShort -> Ptr CShort -> CShort -> Ptr CTHShortStorage -> IO ()

-- |c_THShortVector_cmul : z x y n -> void
foreign import ccall "THVector.h THShortVector_cmul"
  c_THShortVector_cmul :: Ptr CShort -> Ptr CShort -> Ptr CShort -> Ptr CTHShortStorage -> IO ()

-- |c_THShortVector_muls : y x c n -> void
foreign import ccall "THVector.h THShortVector_muls"
  c_THShortVector_muls :: Ptr CShort -> Ptr CShort -> CShort -> Ptr CTHShortStorage -> IO ()

-- |c_THShortVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THShortVector_cdiv"
  c_THShortVector_cdiv :: Ptr CShort -> Ptr CShort -> Ptr CShort -> Ptr CTHShortStorage -> IO ()

-- |c_THShortVector_divs : y x c n -> void
foreign import ccall "THVector.h THShortVector_divs"
  c_THShortVector_divs :: Ptr CShort -> Ptr CShort -> CShort -> Ptr CTHShortStorage -> IO ()

-- |c_THShortVector_copy : y x n -> void
foreign import ccall "THVector.h THShortVector_copy"
  c_THShortVector_copy :: Ptr CShort -> Ptr CShort -> Ptr CTHShortStorage -> IO ()

-- |c_THShortVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THShortVector_vectorDispatchInit"
  c_THShortVector_vectorDispatchInit :: IO ()