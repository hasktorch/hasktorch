{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.Vector where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THCharVector_fill"
  c_fill_ :: Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill = const c_fill_

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THCharVector_cadd"
  c_cadd_ :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd = const c_cadd_

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THCharVector_adds"
  c_adds_ :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- | alias of c_adds_ with unused argument (for CTHState) to unify backpack signatures.
c_adds = const c_adds_

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THCharVector_cmul"
  c_cmul_ :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul = const c_cmul_

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THCharVector_muls"
  c_muls_ :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- | alias of c_muls_ with unused argument (for CTHState) to unify backpack signatures.
c_muls = const c_muls_

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THCharVector_cdiv"
  c_cdiv_ :: Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv = const c_cdiv_

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THCharVector_divs"
  c_divs_ :: Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ()

-- | alias of c_divs_ with unused argument (for CTHState) to unify backpack signatures.
c_divs = const c_divs_

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THCharVector_copy"
  c_copy_ :: Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy = const c_copy_

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THCharVector_normal_fill"
  c_normal_fill_ :: Ptr CChar -> CLLong -> Ptr C'THGenerator -> CChar -> CChar -> IO ()

-- | alias of c_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_fill = const c_normal_fill_

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THCharVector_fill"
  p_fill_ :: FunPtr (Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- | alias of p_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_fill = const p_fill_

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THCharVector_cadd"
  p_cadd_ :: FunPtr (Ptr CChar -> Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- | alias of p_cadd_ with unused argument (for CTHState) to unify backpack signatures.
p_cadd = const p_cadd_

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THCharVector_adds"
  p_adds_ :: FunPtr (Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- | alias of p_adds_ with unused argument (for CTHState) to unify backpack signatures.
p_adds = const p_adds_

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THCharVector_cmul"
  p_cmul_ :: FunPtr (Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- | alias of p_cmul_ with unused argument (for CTHState) to unify backpack signatures.
p_cmul = const p_cmul_

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THCharVector_muls"
  p_muls_ :: FunPtr (Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- | alias of p_muls_ with unused argument (for CTHState) to unify backpack signatures.
p_muls = const p_muls_

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THCharVector_cdiv"
  p_cdiv_ :: FunPtr (Ptr CChar -> Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- | alias of p_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
p_cdiv = const p_cdiv_

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THCharVector_divs"
  p_divs_ :: FunPtr (Ptr CChar -> Ptr CChar -> CChar -> CPtrdiff -> IO ())

-- | alias of p_divs_ with unused argument (for CTHState) to unify backpack signatures.
p_divs = const p_divs_

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THCharVector_copy"
  p_copy_ :: FunPtr (Ptr CChar -> Ptr CChar -> CPtrdiff -> IO ())

-- | alias of p_copy_ with unused argument (for CTHState) to unify backpack signatures.
p_copy = const p_copy_

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THCharVector_normal_fill"
  p_normal_fill_ :: FunPtr (Ptr CChar -> CLLong -> Ptr C'THGenerator -> CChar -> CChar -> IO ())

-- | alias of p_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_normal_fill = const p_normal_fill_