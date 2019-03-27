{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Type.SparseTensorRef.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Type.SparseTensorRef.RawType
import Aten.Type.Tensor.RawType

foreign import ccall safe
               "AtenSparseTensorRef.h SparseTensorRef_newSparseTensorRef"
               c_sparsetensorref_newsparsetensorref ::
               Ptr RawTensor -> IO (Ptr RawSparseTensorRef)
