{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Type.Scalar.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Type.Scalar.RawType

foreign import ccall safe "AtenScalar.h Scalar_newScalar"
               c_scalar_newscalar :: IO (Ptr RawScalar)
