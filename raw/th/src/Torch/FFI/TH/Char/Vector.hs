{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.Vector
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
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THCharVector_fill"
  c_fill :: Ptr (CChar) -> CChar -> CPtrdiff -> IO (())

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THCharVector_cadd"
  c_cadd :: Ptr (CChar) -> Ptr (CChar) -> Ptr (CChar) -> CChar -> CPtrdiff -> IO (())

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THCharVector_adds"
  c_adds :: Ptr (CChar) -> Ptr (CChar) -> CChar -> CPtrdiff -> IO (())

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THCharVector_cmul"
  c_cmul :: Ptr (CChar) -> Ptr (CChar) -> Ptr (CChar) -> CPtrdiff -> IO (())

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THCharVector_muls"
  c_muls :: Ptr (CChar) -> Ptr (CChar) -> CChar -> CPtrdiff -> IO (())

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THCharVector_cdiv"
  c_cdiv :: Ptr (CChar) -> Ptr (CChar) -> Ptr (CChar) -> CPtrdiff -> IO (())

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THCharVector_divs"
  c_divs :: Ptr (CChar) -> Ptr (CChar) -> CChar -> CPtrdiff -> IO (())

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THCharVector_copy"
  c_copy :: Ptr (CChar) -> Ptr (CChar) -> CPtrdiff -> IO (())

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THCharVector_normal_fill"
  c_normal_fill :: Ptr (CChar) -> CLLong -> Ptr (CTHGenerator) -> CChar -> CChar -> IO (())

-- | c_vectorDispatchInit :   -> void
foreign import ccall "THVector.h THCharVector_vectorDispatchInit"
  c_vectorDispatchInit :: IO (())

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THCharVector_fill"
  p_fill :: FunPtr (Ptr (CChar) -> CChar -> CPtrdiff -> IO (()))

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THCharVector_cadd"
  p_cadd :: FunPtr (Ptr (CChar) -> Ptr (CChar) -> Ptr (CChar) -> CChar -> CPtrdiff -> IO (()))

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THCharVector_adds"
  p_adds :: FunPtr (Ptr (CChar) -> Ptr (CChar) -> CChar -> CPtrdiff -> IO (()))

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THCharVector_cmul"
  p_cmul :: FunPtr (Ptr (CChar) -> Ptr (CChar) -> Ptr (CChar) -> CPtrdiff -> IO (()))

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THCharVector_muls"
  p_muls :: FunPtr (Ptr (CChar) -> Ptr (CChar) -> CChar -> CPtrdiff -> IO (()))

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THCharVector_cdiv"
  p_cdiv :: FunPtr (Ptr (CChar) -> Ptr (CChar) -> Ptr (CChar) -> CPtrdiff -> IO (()))

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THCharVector_divs"
  p_divs :: FunPtr (Ptr (CChar) -> Ptr (CChar) -> CChar -> CPtrdiff -> IO (()))

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THCharVector_copy"
  p_copy :: FunPtr (Ptr (CChar) -> Ptr (CChar) -> CPtrdiff -> IO (()))

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THCharVector_normal_fill"
  p_normal_fill :: FunPtr (Ptr (CChar) -> CLLong -> Ptr (CTHGenerator) -> CChar -> CChar -> IO (()))

-- | p_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THCharVector_vectorDispatchInit"
  p_vectorDispatchInit :: FunPtr (IO (()))