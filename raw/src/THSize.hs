{-# LANGUAGE ForeignFunctionInterface #-}

module THSize (
    c_THSize_isSameSizeAs,
    c_THSize_nElement,
    p_THSize_isSameSizeAs,
    p_THSize_nElement) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THSize_isSameSizeAs : sizeA dimsA sizeB dimsB -> int
foreign import ccall unsafe "THSize.h THSize_isSameSizeAs"
  c_THSize_isSameSizeAs :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CInt

-- |c_THSize_nElement : dims size -> ptrdiff_t
foreign import ccall unsafe "THSize.h THSize_nElement"
  c_THSize_nElement :: CLong -> Ptr CLong -> CPtrdiff

-- |p_THSize_isSameSizeAs : Pointer to function sizeA dimsA sizeB dimsB -> int
foreign import ccall unsafe "THSize.h &THSize_isSameSizeAs"
  p_THSize_isSameSizeAs :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CInt)

-- |p_THSize_nElement : Pointer to function dims size -> ptrdiff_t
foreign import ccall unsafe "THSize.h &THSize_nElement"
  p_THSize_nElement :: FunPtr (CLong -> Ptr CLong -> CPtrdiff)