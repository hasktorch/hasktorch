{-# LANGUAGE Strict #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.FFI.THC.State where

import Foreign
import Control.Monad (void)

import Torch.Types.THC (C'THCState)
import qualified Torch.FFI.THC.General as General
import System.IO.Unsafe

-- | calls 'General.c_THCudaShutdown' and 'General.c_THCState_free' from C.
foreign import ccall "&free_CTHState" state_free :: FunPtr (Ptr C'THCState -> IO ())

torchstate :: ForeignPtr C'THCState
torchstate = unsafePerformIO $ do
  s <- General.c_THCState_alloc
  General.c_THCudaInit s
  General.c_THCMagma_init s
  newForeignPtr state_free s
{-# NOINLINE torchstate #-}



