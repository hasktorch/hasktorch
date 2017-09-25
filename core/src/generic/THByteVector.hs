{-# LANGUAGE ForeignFunctionInterface #-}

module THByteVector (
    c_THByteVector_fill,
    c_THByteVector_cadd,
    c_THByteVector_adds,
    c_THByteVector_cmul,
    c_THByteVector_muls,
    c_THByteVector_cdiv,
    c_THByteVector_divs,
    c_THByteVector_copy,
    c_THByteVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteVector_fill : x c n -> void
foreign import ccall "THVector.h THByteVector_fill"
  c_THByteVector_fill :: Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THByteVector_cadd"
  c_THByteVector_cadd :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_adds : y x c n -> void
foreign import ccall "THVector.h THByteVector_adds"
  c_THByteVector_adds :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_cmul : z x y n -> void
foreign import ccall "THVector.h THByteVector_cmul"
  c_THByteVector_cmul :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_muls : y x c n -> void
foreign import ccall "THVector.h THByteVector_muls"
  c_THByteVector_muls :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THByteVector_cdiv"
  c_THByteVector_cdiv :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_divs : y x c n -> void
foreign import ccall "THVector.h THByteVector_divs"
  c_THByteVector_divs :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_copy : y x n -> void
foreign import ccall "THVector.h THByteVector_copy"
  c_THByteVector_copy :: Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- |c_THByteVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THByteVector_vectorDispatchInit"
  c_THByteVector_vectorDispatchInit :: IO ()