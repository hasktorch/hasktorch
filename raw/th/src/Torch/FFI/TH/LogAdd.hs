{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.LogAdd
  ( c_THLogAdd
  , c_THLogSub
  , c_THExpMinusApprox
  , p_THLogAdd
  , p_THLogSub
  , p_THExpMinusApprox
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THLogAdd :  log_a log_b -> double
foreign import ccall "THLogAdd.h c_THLogAdd"
  c_THLogAdd :: CDouble -> CDouble -> IO (CDouble)

-- | c_THLogSub :  log_a log_b -> double
foreign import ccall "THLogAdd.h c_THLogSub"
  c_THLogSub :: CDouble -> CDouble -> IO (CDouble)

-- | c_THExpMinusApprox :  x -> double
foreign import ccall "THLogAdd.h c_THExpMinusApprox"
  c_THExpMinusApprox :: CDouble -> IO (CDouble)

-- | p_THLogAdd : Pointer to function : log_a log_b -> double
foreign import ccall "THLogAdd.h &p_THLogAdd"
  p_THLogAdd :: FunPtr (CDouble -> CDouble -> IO (CDouble))

-- | p_THLogSub : Pointer to function : log_a log_b -> double
foreign import ccall "THLogAdd.h &p_THLogSub"
  p_THLogSub :: FunPtr (CDouble -> CDouble -> IO (CDouble))

-- | p_THExpMinusApprox : Pointer to function : x -> double
foreign import ccall "THLogAdd.h &p_THExpMinusApprox"
  p_THExpMinusApprox :: FunPtr (CDouble -> IO (CDouble))