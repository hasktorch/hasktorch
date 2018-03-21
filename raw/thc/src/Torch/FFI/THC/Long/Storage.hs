{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Long.Storage where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCudaLongStorage_data"
  c_data :: Ptr C'THCState -> Ptr C'THCLongStorage -> IO (Ptr CLong)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCudaLongStorage_size"
  c_size :: Ptr C'THCState -> Ptr C'THCLongStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCudaLongStorage_set"
  c_set :: Ptr C'THCState -> Ptr C'THCLongStorage -> CPtrdiff -> CLong -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCudaLongStorage_get"
  c_get :: Ptr C'THCState -> Ptr C'THCLongStorage -> CPtrdiff -> IO CLong

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCudaLongStorage_new"
  c_new :: Ptr C'THCState -> IO (Ptr C'THCLongStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCudaLongStorage_newWithSize"
  c_newWithSize :: Ptr C'THCState -> CPtrdiff -> IO (Ptr C'THCLongStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCudaLongStorage_newWithSize1"
  c_newWithSize1 :: Ptr C'THCState -> CLong -> IO (Ptr C'THCLongStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCudaLongStorage_newWithSize2"
  c_newWithSize2 :: Ptr C'THCState -> CLong -> CLong -> IO (Ptr C'THCLongStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCudaLongStorage_newWithSize3"
  c_newWithSize3 :: Ptr C'THCState -> CLong -> CLong -> CLong -> IO (Ptr C'THCLongStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCudaLongStorage_newWithSize4"
  c_newWithSize4 :: Ptr C'THCState -> CLong -> CLong -> CLong -> CLong -> IO (Ptr C'THCLongStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCudaLongStorage_newWithMapping"
  c_newWithMapping :: Ptr C'THCState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCLongStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCudaLongStorage_newWithData"
  c_newWithData :: Ptr C'THCState -> Ptr CLong -> CPtrdiff -> IO (Ptr C'THCLongStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCudaLongStorage_setFlag"
  c_setFlag :: Ptr C'THCState -> Ptr C'THCLongStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCudaLongStorage_clearFlag"
  c_clearFlag :: Ptr C'THCState -> Ptr C'THCLongStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCudaLongStorage_retain"
  c_retain :: Ptr C'THCState -> Ptr C'THCLongStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCudaLongStorage_free"
  c_free :: Ptr C'THCState -> Ptr C'THCLongStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCudaLongStorage_resize"
  c_resize :: Ptr C'THCState -> Ptr C'THCLongStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCudaLongStorage_fill"
  c_fill :: Ptr C'THCState -> Ptr C'THCLongStorage -> CLong -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCudaLongStorage_getDevice"
  c_getDevice :: Ptr C'THCState -> Ptr C'THCLongStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCudaLongStorage_data"
  p_data :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> IO (Ptr CLong))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCudaLongStorage_size"
  p_size :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCudaLongStorage_set"
  p_set :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> CPtrdiff -> CLong -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCudaLongStorage_get"
  p_get :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> CPtrdiff -> IO CLong)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCudaLongStorage_new"
  p_new :: FunPtr (Ptr C'THCState -> IO (Ptr C'THCLongStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCudaLongStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr C'THCState -> CPtrdiff -> IO (Ptr C'THCLongStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCudaLongStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr C'THCState -> CLong -> IO (Ptr C'THCLongStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCudaLongStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr C'THCState -> CLong -> CLong -> IO (Ptr C'THCLongStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCudaLongStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr C'THCState -> CLong -> CLong -> CLong -> IO (Ptr C'THCLongStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCudaLongStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr C'THCState -> CLong -> CLong -> CLong -> CLong -> IO (Ptr C'THCLongStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCudaLongStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr C'THCState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr C'THCLongStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCudaLongStorage_newWithData"
  p_newWithData :: FunPtr (Ptr C'THCState -> Ptr CLong -> CPtrdiff -> IO (Ptr C'THCLongStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCudaLongStorage_setFlag"
  p_setFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCudaLongStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCudaLongStorage_retain"
  p_retain :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCudaLongStorage_free"
  p_free :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCudaLongStorage_resize"
  p_resize :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCudaLongStorage_fill"
  p_fill :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> CLong -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCudaLongStorage_getDevice"
  p_getDevice :: FunPtr (Ptr C'THCState -> Ptr C'THCLongStorage -> IO CInt)