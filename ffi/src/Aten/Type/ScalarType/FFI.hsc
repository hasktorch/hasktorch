{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Type.ScalarType.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Type.ScalarType.RawType

foreign import ccall safe
               "AtenScalarType.h ScalarType_newScalarType"
               c_scalartype_newscalartype :: IO (Ptr RawScalarType)
