{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Tensor.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Tensor.RawType

foreign import ccall safe "AtenTensor.h Tensor_newTensor"
               c_tensor_newtensor :: IO (Ptr RawTensor)
