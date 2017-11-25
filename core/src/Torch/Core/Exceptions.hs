{-# LANGUAGE ForeignFunctionInterface #-}

module Torch.Core.Exceptions
  ( TorchException(..)
  , module X
  , p_errorHandler
  , p_argErrorHandler
  ) where

import Control.Exception.Safe as X
-- import Control.Exception.Base as X (catch)
import Data.Typeable (Typeable)
import Data.Text (Text)

import Foreign
import Foreign.C.String
import Foreign.C.Types
import THTypes

data TorchException
  = MathException Text
  deriving (Show, Typeable)

instance Exception TorchException

foreign import ccall unsafe "error_handler.h testFunction"
  c_testFunction :: IO ()

foreign import ccall unsafe "error_handler.h &testFunction"
  p_testFunction :: FunPtr (IO ())

foreign import ccall unsafe "error_handler.h errorHandler"
  c_errorHandler :: CString -> IO ()

foreign import ccall unsafe "error_handler.h &errorHandler"
  p_errorHandler :: FunPtr (CString -> IO ())

foreign import ccall unsafe "error_handler.h argErrorHandler"
  c_argErrorHandler :: CString -> IO ()

foreign import ccall unsafe "error_handler.h &argErrorHandler"
  p_argErrorHandler :: FunPtr (CString -> IO ())

test = do
  c_testFunction
