{-# LANGUAGE ForeignFunctionInterface #-}
module THByteVector
  ( c_fill
  , c_cadd
  , c_adds
  , c_cmul
  , c_muls
  , c_cdiv
  , c_divs
  , c_copy
  , c_normal_fill
  , c_vectorDispatchInit
  , p_fill
  , p_cadd
  , p_adds
  , p_cmul
  , p_muls
  , p_cdiv
  , p_divs
  , p_copy
  , p_normal_fill
  , p_vectorDispatchInit
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THByteVector_fill"
  c_fill :: Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THByteVector_cadd"
  c_cadd :: Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THByteVector_adds"
  c_adds :: Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THByteVector_cmul"
  c_cmul :: Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THByteVector_muls"
  c_muls :: Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THByteVector_cdiv"
  c_cdiv :: Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THByteVector_divs"
  c_divs :: Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THByteVector_copy"
  c_copy :: Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THByteVector_normal_fill"
  c_normal_fill :: Ptr CUChar -> CLLong -> Ptr CTHGenerator -> CUChar -> CUChar -> IO ()

-- | c_vectorDispatchInit :   -> void
foreign import ccall "THVector.h THByteVector_vectorDispatchInit"
  c_vectorDispatchInit :: IO ()

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THByteVector_fill"
  p_fill :: FunPtr (Ptr CUChar -> CUChar -> CPtrdiff -> IO ())

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THByteVector_cadd"
  p_cadd :: FunPtr (Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ())

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THByteVector_adds"
  p_adds :: FunPtr (Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ())

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THByteVector_cmul"
  p_cmul :: FunPtr (Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ())

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THByteVector_muls"
  p_muls :: FunPtr (Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ())

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THByteVector_cdiv"
  p_cdiv :: FunPtr (Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ())

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THByteVector_divs"
  p_divs :: FunPtr (Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ())

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THByteVector_copy"
  p_copy :: FunPtr (Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ())

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THByteVector_normal_fill"
  p_normal_fill :: FunPtr (Ptr CUChar -> CLLong -> Ptr CTHGenerator -> CUChar -> CUChar -> IO ())

-- | p_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THByteVector_vectorDispatchInit"
  p_vectorDispatchInit :: FunPtr (IO ())