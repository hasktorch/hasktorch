{-# LANGUAGE ForeignFunctionInterface #-}

module THLogAdd (
    c_THLogAdd,
    c_THLogSub,
    c_THExpMinusApprox,
    p_THLogAdd,
    p_THLogSub,
    p_THExpMinusApprox) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THLogAdd : log_a log_b -> double
foreign import ccall "THLogAdd.h THLogAdd"
  c_THLogAdd :: CDouble -> CDouble -> CDouble

-- |c_THLogSub : log_a log_b -> double
foreign import ccall "THLogAdd.h THLogSub"
  c_THLogSub :: CDouble -> CDouble -> CDouble

-- |c_THExpMinusApprox : x -> double
foreign import ccall "THLogAdd.h THExpMinusApprox"
  c_THExpMinusApprox :: CDouble -> CDouble

-- |p_THLogAdd : Pointer to function : log_a log_b -> double
foreign import ccall "THLogAdd.h &THLogAdd"
  p_THLogAdd :: FunPtr (CDouble -> CDouble -> CDouble)

-- |p_THLogSub : Pointer to function : log_a log_b -> double
foreign import ccall "THLogAdd.h &THLogSub"
  p_THLogSub :: FunPtr (CDouble -> CDouble -> CDouble)

-- |p_THExpMinusApprox : Pointer to function : x -> double
foreign import ccall "THLogAdd.h &THExpMinusApprox"
  p_THExpMinusApprox :: FunPtr (CDouble -> CDouble)
