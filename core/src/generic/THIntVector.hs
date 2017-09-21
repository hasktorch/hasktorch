{-# LANGUAGE ForeignFunctionInterface#-}

module THIntVector (
    c_THIntVector_fill,
    c_THIntVector_cadd,
    c_THIntVector_adds,
    c_THIntVector_cmul,
    c_THIntVector_muls,
    c_THIntVector_cdiv,
    c_THIntVector_divs,
    c_THIntVector_copy,
    c_THIntVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntVector_fill : x c n -> void
foreign import ccall "THVector.h THIntVector_fill"
  c_THIntVector_fill :: Ptr CInt -> CInt -> Ptr CTHIntStorage -> IO ()

-- |c_THIntVector_cadd : z x y c n -> void
foreign import ccall "THVector.h THIntVector_cadd"
  c_THIntVector_cadd :: Ptr CInt -> Ptr CInt -> Ptr CInt -> CInt -> Ptr CTHIntStorage -> IO ()

-- |c_THIntVector_adds : y x c n -> void
foreign import ccall "THVector.h THIntVector_adds"
  c_THIntVector_adds :: Ptr CInt -> Ptr CInt -> CInt -> Ptr CTHIntStorage -> IO ()

-- |c_THIntVector_cmul : z x y n -> void
foreign import ccall "THVector.h THIntVector_cmul"
  c_THIntVector_cmul :: Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr CTHIntStorage -> IO ()

-- |c_THIntVector_muls : y x c n -> void
foreign import ccall "THVector.h THIntVector_muls"
  c_THIntVector_muls :: Ptr CInt -> Ptr CInt -> CInt -> Ptr CTHIntStorage -> IO ()

-- |c_THIntVector_cdiv : z x y n -> void
foreign import ccall "THVector.h THIntVector_cdiv"
  c_THIntVector_cdiv :: Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr CTHIntStorage -> IO ()

-- |c_THIntVector_divs : y x c n -> void
foreign import ccall "THVector.h THIntVector_divs"
  c_THIntVector_divs :: Ptr CInt -> Ptr CInt -> CInt -> Ptr CTHIntStorage -> IO ()

-- |c_THIntVector_copy : y x n -> void
foreign import ccall "THVector.h THIntVector_copy"
  c_THIntVector_copy :: Ptr CInt -> Ptr CInt -> Ptr CTHIntStorage -> IO ()

-- |c_THIntVector_vectorDispatchInit :  -> void
foreign import ccall "THVector.h THIntVector_vectorDispatchInit"
  c_THIntVector_vectorDispatchInit :: IO ()