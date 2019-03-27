{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Type.Tensor.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Type.Tensor.RawType

foreign import ccall safe "AtenTensor.h Tensor_newTensor"
               c_tensor_newtensor :: IO (Ptr RawTensor)

foreign import ccall safe "AtenTensor.h Tensor_tensor_dim"
               c_tensor_tensor_dim :: Ptr RawTensor -> IO CLong

foreign import ccall safe
               "AtenTensor.h Tensor_tensor_storage_offset"
               c_tensor_tensor_storage_offset :: Ptr RawTensor -> IO CLong

foreign import ccall safe "AtenTensor.h Tensor_tensor_defined"
               c_tensor_tensor_defined :: Ptr RawTensor -> IO CInt

foreign import ccall safe "AtenTensor.h Tensor_tensor_reset"
               c_tensor_tensor_reset :: Ptr RawTensor -> IO ()

foreign import ccall safe "AtenTensor.h Tensor_tensor_cpu"
               c_tensor_tensor_cpu :: Ptr RawTensor -> IO ()

foreign import ccall safe "AtenTensor.h Tensor_tensor_cuda"
               c_tensor_tensor_cuda :: Ptr RawTensor -> IO ()

foreign import ccall safe "AtenTensor.h Tensor_tensor_print"
               c_tensor_tensor_print :: Ptr RawTensor -> IO ()
