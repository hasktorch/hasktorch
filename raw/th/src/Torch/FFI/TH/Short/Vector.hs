{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.Vector
  ( c_fill
  , c_cadd
  , c_adds
  , c_cmul
  , c_muls
  , c_cdiv
  , c_divs
  , c_copy
  , c_neg
  , c_normal_fill
  , c_abs
  , c_vectorDispatchInit
  , p_fill
  , p_cadd
  , p_adds
  , p_cmul
  , p_muls
  , p_cdiv
  , p_divs
  , p_copy
  , p_neg
  , p_normal_fill
  , p_abs
  , p_vectorDispatchInit
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THShortVector_fill"
  c_fill :: Ptr (CShort) -> CShort -> CPtrdiff -> IO (())

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THShortVector_cadd"
  c_cadd :: Ptr (CShort) -> Ptr (CShort) -> Ptr (CShort) -> CShort -> CPtrdiff -> IO (())

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THShortVector_adds"
  c_adds :: Ptr (CShort) -> Ptr (CShort) -> CShort -> CPtrdiff -> IO (())

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THShortVector_cmul"
  c_cmul :: Ptr (CShort) -> Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (())

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THShortVector_muls"
  c_muls :: Ptr (CShort) -> Ptr (CShort) -> CShort -> CPtrdiff -> IO (())

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THShortVector_cdiv"
  c_cdiv :: Ptr (CShort) -> Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (())

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THShortVector_divs"
  c_divs :: Ptr (CShort) -> Ptr (CShort) -> CShort -> CPtrdiff -> IO (())

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THShortVector_copy"
  c_copy :: Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (())

-- | c_neg :  y x n -> void
foreign import ccall "THVector.h THShortVector_neg"
  c_neg :: Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (())

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THShortVector_normal_fill"
  c_normal_fill :: Ptr (CShort) -> CLLong -> Ptr (CTHGenerator) -> CShort -> CShort -> IO (())

-- | c_abs :  y x n -> void
foreign import ccall "THVector.h THShortVector_abs"
  c_abs :: Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (())

-- | c_vectorDispatchInit :   -> void
foreign import ccall "THVector.h THShortVector_vectorDispatchInit"
  c_vectorDispatchInit :: IO (())

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THShortVector_fill"
  p_fill :: FunPtr (Ptr (CShort) -> CShort -> CPtrdiff -> IO (()))

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THShortVector_cadd"
  p_cadd :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> Ptr (CShort) -> CShort -> CPtrdiff -> IO (()))

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THShortVector_adds"
  p_adds :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> CShort -> CPtrdiff -> IO (()))

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THShortVector_cmul"
  p_cmul :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (()))

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THShortVector_muls"
  p_muls :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> CShort -> CPtrdiff -> IO (()))

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THShortVector_cdiv"
  p_cdiv :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (()))

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THShortVector_divs"
  p_divs :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> CShort -> CPtrdiff -> IO (()))

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THShortVector_copy"
  p_copy :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (()))

-- | p_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THShortVector_neg"
  p_neg :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (()))

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THShortVector_normal_fill"
  p_normal_fill :: FunPtr (Ptr (CShort) -> CLLong -> Ptr (CTHGenerator) -> CShort -> CShort -> IO (()))

-- | p_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THShortVector_abs"
  p_abs :: FunPtr (Ptr (CShort) -> Ptr (CShort) -> CPtrdiff -> IO (()))

-- | p_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THShortVector_vectorDispatchInit"
  p_vectorDispatchInit :: FunPtr (IO (()))