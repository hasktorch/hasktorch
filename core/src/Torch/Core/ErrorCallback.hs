{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.Core.ErrorCallback where

import Foreign.C.Types

-- |error function to be called from C++
error_hs :: IO ()
error_hs = error "A torch error occurred"

foreign export ccall error_hs :: IO ()
