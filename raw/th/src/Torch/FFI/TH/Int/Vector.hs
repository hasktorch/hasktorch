{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.Vector where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THIntVector_fill"
  c_fill :: Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THIntVector_cadd"
  c_cadd :: Ptr CInt -> Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THIntVector_adds"
  c_adds :: Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THIntVector_cmul"
  c_cmul :: Ptr CInt -> Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THIntVector_muls"
  c_muls :: Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THIntVector_cdiv"
  c_cdiv :: Ptr CInt -> Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THIntVector_divs"
  c_divs :: Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ()

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THIntVector_copy"
  c_copy :: Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- | c_neg :  y x n -> void
foreign import ccall "THVector.h THIntVector_neg"
  c_neg :: Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THIntVector_normal_fill"
  c_normal_fill :: Ptr CInt -> CLLong -> Ptr CTHGenerator -> CInt -> CInt -> IO ()

-- | c_abs :  y x n -> void
foreign import ccall "THVector.h THIntVector_abs"
  c_abs :: Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ()

-- | c_vectorDispatchInit :   -> void
foreign import ccall "THVector.h THIntVector_vectorDispatchInit"
  c_vectorDispatchInit :: IO ()

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THIntVector_fill"
  p_fill :: FunPtr (Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THIntVector_cadd"
  p_cadd :: FunPtr (Ptr CInt -> Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THIntVector_adds"
  p_adds :: FunPtr (Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THIntVector_cmul"
  p_cmul :: FunPtr (Ptr CInt -> Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THIntVector_muls"
  p_muls :: FunPtr (Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THIntVector_cdiv"
  p_cdiv :: FunPtr (Ptr CInt -> Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THIntVector_divs"
  p_divs :: FunPtr (Ptr CInt -> Ptr CInt -> CInt -> CPtrdiff -> IO ())

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THIntVector_copy"
  p_copy :: FunPtr (Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- | p_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THIntVector_neg"
  p_neg :: FunPtr (Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THIntVector_normal_fill"
  p_normal_fill :: FunPtr (Ptr CInt -> CLLong -> Ptr CTHGenerator -> CInt -> CInt -> IO ())

-- | p_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THIntVector_abs"
  p_abs :: FunPtr (Ptr CInt -> Ptr CInt -> CPtrdiff -> IO ())

-- | p_vectorDispatchInit : Pointer to function :  -> void
foreign import ccall "THVector.h &THIntVector_vectorDispatchInit"
  p_vectorDispatchInit :: FunPtr (IO ())