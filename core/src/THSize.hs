{-# LANGUAGE ForeignFunctionInterface #-}

module THSize (
    c_THSize_isSameSizeAs,
    c_THSize_nElement) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THSize_isSameSizeAs : sizeA dimsA sizeB dimsB -> int
foreign import ccall "THSize.h THSize_isSameSizeAs"
  c_THSize_isSameSizeAs :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CInt

-- |c_THSize_nElement : dims size -> ptrdiff_t
foreign import ccall "THSize.h THSize_nElement"
  c_THSize_nElement :: CLong -> Ptr CLong -> CPtrdiff