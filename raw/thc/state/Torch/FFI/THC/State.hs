module Torch.FFI.THC.State where

import Foreign hiding (void)
import Control.Monad (void)

import Torch.Types.THC (C'THCState)
import qualified Torch.FFI.THC.General as General

newCState :: IO (Ptr C'THCState)
newCState = do
  s <- General.c_THCState_alloc
  General.c_THCudaInit s
  pure s


stopState :: Ptr C'THCState -> IO ()
stopState s = void $ do
  -- FIXME: This next line throws a c-level error...? If so comment out "cuda shutdown" for now and fix this later
  General.c_THCudaShutdown s
  General.c_THCState_free s

-- FIXME
-- manageState :: Ptr C'THCState -> IO (ForeignPtr C'THCState)
-- manageState = newForeignPtr (const $ pure ()) --  General.p_THCState_free

foreign import ccall "&free_CTHState" state_free :: FunPtr (Ptr C'THCState -> IO ())

manageState :: Ptr C'THCState -> IO (ForeignPtr C'THCState)
manageState = newForeignPtr state_free


shutdown :: ForeignPtr C'THCState -> IO ()
shutdown fp = withForeignPtr fp General.c_THCudaShutdown

