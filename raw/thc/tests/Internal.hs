module Internal where

import Foreign
import Test.Hspec

import Torch.Types.THC (C'THCState)
import qualified Torch.FFI.THC.General as General

withCudaState :: (Ptr C'THCState -> Spec) -> Spec
withCudaState fn = do
  g <- runIO General.c_THCState_alloc
  fn g
  runIO $ General.c_THCState_free g



