module Internal where

import Foreign hiding (void)
import Test.Hspec
import Control.Monad (void)

import Torch.Types.THC (C'THCState)
import qualified Torch.FFI.THC.General as General

cudaState :: IO (Ptr C'THCState)
cudaState = do
  s <- General.c_THCState_alloc
  General.c_THCudaInit s
  pure s

stopCuda :: Ptr C'THCState -> IO ()
stopCuda s = void $ do
  General.c_THCState_free s

  -- FIXME: This throws a segfault...
  -- General.c_THCudaShutdown s


