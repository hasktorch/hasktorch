{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.Vector where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THByteVector_fill"
  c_fill_ :: Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill :: Ptr C'THState -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()
c_fill = const c_fill_

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THByteVector_cadd"
  c_cadd_ :: Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd :: Ptr C'THState -> Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()
c_cadd = const c_cadd_

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THByteVector_adds"
  c_adds_ :: Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | alias of c_adds_ with unused argument (for CTHState) to unify backpack signatures.
c_adds :: Ptr C'THState -> Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()
c_adds = const c_adds_

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THByteVector_cmul"
  c_cmul_ :: Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul :: Ptr C'THState -> Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()
c_cmul = const c_cmul_

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THByteVector_muls"
  c_muls_ :: Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | alias of c_muls_ with unused argument (for CTHState) to unify backpack signatures.
c_muls :: Ptr C'THState -> Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()
c_muls = const c_muls_

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THByteVector_cdiv"
  c_cdiv_ :: Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv :: Ptr C'THState -> Ptr CUChar -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()
c_cdiv = const c_cdiv_

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THByteVector_divs"
  c_divs_ :: Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()

-- | alias of c_divs_ with unused argument (for CTHState) to unify backpack signatures.
c_divs :: Ptr C'THState -> Ptr CUChar -> Ptr CUChar -> CUChar -> CPtrdiff -> IO ()
c_divs = const c_divs_

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THByteVector_copy"
  c_copy_ :: Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy :: Ptr C'THState -> Ptr CUChar -> Ptr CUChar -> CPtrdiff -> IO ()
c_copy = const c_copy_

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THByteVector_normal_fill"
  c_normal_fill_ :: Ptr CUChar -> CLLong -> Ptr C'THGenerator -> CUChar -> CUChar -> IO ()

-- | alias of c_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_fill :: Ptr C'THState -> Ptr CUChar -> CLLong -> Ptr C'THGenerator -> CUChar -> CUChar -> IO ()
c_normal_fill = const c_normal_fill_

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
  p_normal_fill :: FunPtr (Ptr CUChar -> CLLong -> Ptr C'THGenerator -> CUChar -> CUChar -> IO ())