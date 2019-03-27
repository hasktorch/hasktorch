{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Type.TensorOptions.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Type.TensorOptions.RawType

foreign import ccall safe
               "AtenTensorOptions.h TensorOptions_newTensorOptions"
               c_tensoroptions_newtensoroptions :: IO (Ptr RawTensorOptions)
