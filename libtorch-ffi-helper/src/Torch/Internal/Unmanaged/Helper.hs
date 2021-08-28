{-# LANGUAGE ForeignFunctionInterface #-}

module Torch.Internal.Unmanaged.Helper where

import Foreign.Ptr

foreign import ccall "wrapper" callbackHelper :: (Ptr () -> IO (Ptr ())) -> IO (FunPtr (Ptr () -> IO (Ptr ())))

foreign import ccall "wrapper" callbackHelper2 :: (Ptr () -> Ptr () -> IO (Ptr ())) -> IO (FunPtr (Ptr () -> Ptr () -> IO (Ptr ())))
