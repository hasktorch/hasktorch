{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.Vector where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_fill :  x c n -> void
foreign import ccall "THVector.h THShortVector_fill"
  c_fill_ :: Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- | alias of c_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_fill :: Ptr C'THState -> Ptr CShort -> CShort -> CPtrdiff -> IO ()
c_fill = const c_fill_

-- | c_cadd :  z x y c n -> void
foreign import ccall "THVector.h THShortVector_cadd"
  c_cadd_ :: Ptr CShort -> Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- | alias of c_cadd_ with unused argument (for CTHState) to unify backpack signatures.
c_cadd :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()
c_cadd = const c_cadd_

-- | c_adds :  y x c n -> void
foreign import ccall "THVector.h THShortVector_adds"
  c_adds_ :: Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- | alias of c_adds_ with unused argument (for CTHState) to unify backpack signatures.
c_adds :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()
c_adds = const c_adds_

-- | c_cmul :  z x y n -> void
foreign import ccall "THVector.h THShortVector_cmul"
  c_cmul_ :: Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()

-- | alias of c_cmul_ with unused argument (for CTHState) to unify backpack signatures.
c_cmul :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()
c_cmul = const c_cmul_

-- | c_muls :  y x c n -> void
foreign import ccall "THVector.h THShortVector_muls"
  c_muls_ :: Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- | alias of c_muls_ with unused argument (for CTHState) to unify backpack signatures.
c_muls :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()
c_muls = const c_muls_

-- | c_cdiv :  z x y n -> void
foreign import ccall "THVector.h THShortVector_cdiv"
  c_cdiv_ :: Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()

-- | alias of c_cdiv_ with unused argument (for CTHState) to unify backpack signatures.
c_cdiv :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()
c_cdiv = const c_cdiv_

-- | c_divs :  y x c n -> void
foreign import ccall "THVector.h THShortVector_divs"
  c_divs_ :: Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- | alias of c_divs_ with unused argument (for CTHState) to unify backpack signatures.
c_divs :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()
c_divs = const c_divs_

-- | c_copy :  y x n -> void
foreign import ccall "THVector.h THShortVector_copy"
  c_copy_ :: Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()
c_copy = const c_copy_

-- | c_neg :  y x n -> void
foreign import ccall "THVector.h THShortVector_neg"
  c_neg_ :: Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()

-- | alias of c_neg_ with unused argument (for CTHState) to unify backpack signatures.
c_neg :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()
c_neg = const c_neg_

-- | c_normal_fill :  data size generator mean stddev -> void
foreign import ccall "THVector.h THShortVector_normal_fill"
  c_normal_fill_ :: Ptr CShort -> CLLong -> Ptr C'THGenerator -> CShort -> CShort -> IO ()

-- | alias of c_normal_fill_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_fill :: Ptr C'THState -> Ptr CShort -> CLLong -> Ptr C'THGenerator -> CShort -> CShort -> IO ()
c_normal_fill = const c_normal_fill_

-- | c_abs :  y x n -> void
foreign import ccall "THVector.h THShortVector_abs"
  c_abs_ :: Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()

-- | alias of c_abs_ with unused argument (for CTHState) to unify backpack signatures.
c_abs :: Ptr C'THState -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()
c_abs = const c_abs_

-- | p_fill : Pointer to function : x c n -> void
foreign import ccall "THVector.h &THShortVector_fill"
  p_fill :: FunPtr (Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- | p_cadd : Pointer to function : z x y c n -> void
foreign import ccall "THVector.h &THShortVector_cadd"
  p_cadd :: FunPtr (Ptr CShort -> Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- | p_adds : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THShortVector_adds"
  p_adds :: FunPtr (Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- | p_cmul : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THShortVector_cmul"
  p_cmul :: FunPtr (Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ())

-- | p_muls : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THShortVector_muls"
  p_muls :: FunPtr (Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- | p_cdiv : Pointer to function : z x y n -> void
foreign import ccall "THVector.h &THShortVector_cdiv"
  p_cdiv :: FunPtr (Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ())

-- | p_divs : Pointer to function : y x c n -> void
foreign import ccall "THVector.h &THShortVector_divs"
  p_divs :: FunPtr (Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- | p_copy : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THShortVector_copy"
  p_copy :: FunPtr (Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ())

-- | p_neg : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THShortVector_neg"
  p_neg :: FunPtr (Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ())

-- | p_normal_fill : Pointer to function : data size generator mean stddev -> void
foreign import ccall "THVector.h &THShortVector_normal_fill"
  p_normal_fill :: FunPtr (Ptr CShort -> CLLong -> Ptr C'THGenerator -> CShort -> CShort -> IO ())

-- | p_abs : Pointer to function : y x n -> void
foreign import ccall "THVector.h &THShortVector_abs"
  p_abs :: FunPtr (Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ())