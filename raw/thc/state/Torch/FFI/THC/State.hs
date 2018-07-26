module Torch.FFI.THC.State where

import Foreign hiding (void)
import Control.Monad (void)

import Torch.Types.THC (C'THCState)
import qualified Torch.FFI.THC.General as General
import System.IO.Unsafe

newCState :: IO (Ptr C'THCState)
newCState = pure pureCState

pureCState :: Ptr C'THCState
pureCState = unsafeDupablePerformIO mkNewCState

mkNewCState :: IO (Ptr C'THCState)
mkNewCState = do
  s <- General.c_THCState_alloc
  General.c_THCudaInit s
  General.c_THCMagma_init s
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

cstate :: ForeignPtr C'THCState
cstate = unsafeDupablePerformIO $ mkNewCState >>= newForeignPtr state_free


