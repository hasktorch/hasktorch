{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.Vector where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THHalfVector_fill"
  c_fill_ :: Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill = const c_fill_

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THHalfVector_cadd"
  c_cadd_ :: Ptr CTHHalf -> Ptr CTHHalf -> Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd = const c_cadd_

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THHalfVector_adds"
  c_adds_ :: Ptr CTHHalf -> Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ()

-- | alias of c_adds_ with unused argument (for CTHState) to unify backpack signatures.
c_adds = const c_adds_

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THHalfVector_cmul"
  c_cmul_ :: Ptr CTHHalf -> Ptr CTHHalf -> Ptr CTHHalf -> CPtrdiff -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul = const c_cmul_

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THHalfVector_muls"
  c_muls_ :: Ptr CTHHalf -> Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ()

-- | alias of c_muls_ with unused argument (for CTHState) to unify backpack signatures.
c_muls = const c_muls_

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THHalfVector_cdiv"
  c_cdiv_ :: Ptr CTHHalf -> Ptr CTHHalf -> Ptr CTHHalf -> CPtrdiff -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv = const c_cdiv_

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THHalfVector_divs"
  c_divs_ :: Ptr CTHHalf -> Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ()

-- | alias of c_divs_ with unused argument (for CTHState) to unify backpack signatures.
c_divs = const c_divs_

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THHalfVector_copy"
  c_copy_ :: Ptr CTHHalf -> Ptr CTHHalf -> CPtrdiff -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy = const c_copy_

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THHalfVector_normal_fill"
  c_normal_fill_ :: Ptr CTHHalf -> CLLong -> Ptr C'THGenerator -> CTHHalf -> CTHHalf -> IO ()

-- | alias of c_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_fill = const c_normal_fill_

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THHalfVector_fill"
  p_fill_ :: FunPtr (Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ())

-- | alias of p_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_fill = const p_fill_

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THHalfVector_cadd"
  p_cadd_ :: FunPtr (Ptr CTHHalf -> Ptr CTHHalf -> Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ())

-- | alias of p_cadd_ with unused argument (for CTHState) to unify backpack signatures.
p_cadd = const p_cadd_

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THHalfVector_adds"
  p_adds_ :: FunPtr (Ptr CTHHalf -> Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ())

-- | alias of p_adds_ with unused argument (for CTHState) to unify backpack signatures.
p_adds = const p_adds_

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THHalfVector_cmul"
  p_cmul_ :: FunPtr (Ptr CTHHalf -> Ptr CTHHalf -> Ptr CTHHalf -> CPtrdiff -> IO ())

-- | alias of p_cmul_ with unused argument (for CTHState) to unify backpack signatures.
p_cmul = const p_cmul_

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THHalfVector_muls"
  p_muls_ :: FunPtr (Ptr CTHHalf -> Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ())

-- | alias of p_muls_ with unused argument (for CTHState) to unify backpack signatures.
p_muls = const p_muls_

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THHalfVector_cdiv"
  p_cdiv_ :: FunPtr (Ptr CTHHalf -> Ptr CTHHalf -> Ptr CTHHalf -> CPtrdiff -> IO ())

-- | alias of p_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
p_cdiv = const p_cdiv_

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THHalfVector_divs"
  p_divs_ :: FunPtr (Ptr CTHHalf -> Ptr CTHHalf -> CTHHalf -> CPtrdiff -> IO ())

-- | alias of p_divs_ with unused argument (for CTHState) to unify backpack signatures.
p_divs = const p_divs_

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THHalfVector_copy"
  p_copy_ :: FunPtr (Ptr CTHHalf -> Ptr CTHHalf -> CPtrdiff -> IO ())

-- | alias of p_copy_ with unused argument (for CTHState) to unify backpack signatures.
p_copy = const p_copy_

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THHalfVector_normal_fill"
  p_normal_fill_ :: FunPtr (Ptr CTHHalf -> CLLong -> Ptr C'THGenerator -> CTHHalf -> CTHHalf -> IO ())

-- | alias of p_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
p_normal_fill = const p_normal_fill_