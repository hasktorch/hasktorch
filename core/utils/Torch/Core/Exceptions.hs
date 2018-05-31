-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Core.Exceptions
-- Copyright :  (c) Hasktorch devs 2017
-- License   :  BSD3
-- Maintainer:  Sam Stites <sam@stites.io>
-- Stability :  experimental
-- Portability: non-portable
--
-- Package to start off hasktorch exception handling.
--
-- TODO: Move this into a seperate package so that this can be used by
-- 'hasktorch-classes'
-------------------------------------------------------------------------------
{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.Core.Exceptions
  ( TorchException(..)
  , module X
  {-
  , c_testHasktorchLib
  , p_testHasktorchLib
  , c_errorHandler
  , p_errorHandler
  , c_argErrorHandler
  , p_argErrorHandler
  , c_THSetErrorHandler
  -}
  ) where

import Control.Exception.Safe as X
-- import Control.Exception.Base as X (catch)
import Data.Typeable (Typeable)
import Data.Text (Text)

import Foreign
import Foreign.C.String

-- | The base Torch exception class
data TorchException
  = MathException Text
  deriving (Show, Typeable)

instance Exception TorchException

{- Hasktorch error handler -}
{-
foreign import ccall unsafe "error_handler.h testFunction"
  c_testHasktorchLib :: IO ()

foreign import ccall unsafe "error_handler.h &testFunction"
  p_testHasktorchLib :: FunPtr (IO ())

foreign import ccall unsafe "error_handler.h errorHandler"
  c_errorHandler :: CString -> IO ()

foreign import ccall unsafe "error_handler.h &errorHandler"
  p_errorHandler :: FunPtr (CString -> IO ())

foreign import ccall unsafe "error_handler.h argErrorHandler"
  c_argErrorHandler :: CString -> IO ()

foreign import ccall unsafe "error_handler.h &argErrorHandler"
  p_argErrorHandler :: FunPtr (CString -> IO ())

{- THGeneral options to configure error handler -}


-- TH_API void THSetErrorHandler(THErrorHandlerFunction new_handler, void *data);
foreign import ccall "THGeneral.h.in THSetErrorHandler"
  c_THSetErrorHandler :: FunPtr (CString -> IO ()) -> IO ()
-}
{-
-- TH_API double THLog1p(const double x);
foreign import ccall unsafe "THGeneral.h.in THLog1p"
  c_THLog1p :: CDouble -> CDouble

-- safe version of potrf
-- |c_Torch.FFI.TH.Double.Tensor_potrf : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h Torch.FFI.TH.Double.Tensor_potrf"
  c_safe_Torch.FFI.TH.Double.Tensor_potrf :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ()

lapackTest :: IO ()
lapackTest = do
  putStrLn "Setting error handler"
  c_THSetErrorHandler p_errorHandler
  putStrLn "Cholesky decomposition should fail:"
  opt <- newCString "U"
  dims <- Dim.someDimsM [2, 2]
  a <- constant' dims 2

  Gen.c_set2d a 0 0 1.0
  Gen.c_set2d a 0 1 0.0
  Gen.c_set2d a 1 1 (-1.0)
  Gen.c_set2d a 1 0 0.0
  resA <- constant' dims 5.0
  dispRaw a
  c_safe_Torch.FFI.TH.Double.Tensor_potrf resA a opt
  dispRaw a
  -- dispRaw resA -- TODO: what should happen when potrf has an error
  c_Torch.FFI.TH.Double.Tensor_free a
  c_Torch.FFI.TH.Double.Tensor_free resA
  pure ()

test = do
  c_testHasktorchLib
  c_THSetErrorHandler p_errorHandler
  lapackTest
  putStrLn "Done"
  -}
