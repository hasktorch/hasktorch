module Torch.Internal.Unmanaged.Helper where

import Foreign.Ptr

foreign import ccall "wrapper" callbackHelper :: (Ptr () -> IO (Ptr ())) -> IO (FunPtr (Ptr () -> IO (Ptr ())))
