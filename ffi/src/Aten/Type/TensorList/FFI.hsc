{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Type.TensorList.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Type.TensorList.RawType

foreign import ccall safe
               "AtenTensorList.h TensorList_newTensorList"
               c_tensorlist_newtensorlist :: IO (Ptr RawTensorList)
