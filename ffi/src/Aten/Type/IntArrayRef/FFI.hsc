{-# LANGUAGE ForeignFunctionInterface #-}
module Aten.Type.IntArrayRef.FFI where
import Foreign.C
import Foreign.Ptr
import Aten.Type.IntArrayRef.RawType

foreign import ccall safe
               "AtenIntArrayRef.h IntArrayRef_newIntArrayRef"
               c_intarrayref_newintarrayref :: IO (Ptr RawIntArrayRef)
