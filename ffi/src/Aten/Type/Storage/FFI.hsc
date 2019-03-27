{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Type.Storage.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Type.Storage.RawType

foreign import ccall safe "AtenStorage.h Storage_newStorage"
               c_storage_newstorage :: IO (Ptr RawStorage)
