{-# LANGUAGE ForeignFunctionInterface #-}

module THLogAdd (
    c_THLogAdd,
    c_THLogSub,
    c_THExpMinusApprox) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLogAdd : log_a log_b -> double
foreign import ccall "THLogAdd.h THLogAdd"
  c_THLogAdd :: CDouble -> CDouble -> CDouble

-- |c_THLogSub : log_a log_b -> double
foreign import ccall "THLogAdd.h THLogSub"
  c_THLogSub :: CDouble -> CDouble -> CDouble

-- |c_THExpMinusApprox : x -> double
foreign import ccall "THLogAdd.h THExpMinusApprox"
  c_THExpMinusApprox :: CDouble -> CDouble