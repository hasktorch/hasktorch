{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Long.Storage
  ( c_data
  , c_size
  , c_set
  , c_get
  , c_new
  , c_newWithSize
  , c_newWithSize1
  , c_newWithSize2
  , c_newWithSize3
  , c_newWithSize4
  , c_newWithMapping
  , c_newWithData
  , c_setFlag
  , c_clearFlag
  , c_retain
  , c_free
  , c_resize
  , c_fill
  , c_getDevice
  , p_data
  , p_size
  , p_set
  , p_get
  , p_new
  , p_newWithSize
  , p_newWithSize1
  , p_newWithSize2
  , p_newWithSize3
  , p_newWithSize4
  , p_newWithMapping
  , p_newWithData
  , p_setFlag
  , p_clearFlag
  , p_retain
  , p_free
  , p_resize
  , p_fill
  , p_getDevice
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :  state  -> real *
foreign import ccall "THCStorage.h THCLongStorage_data"
  c_data :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO (Ptr CLong)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCLongStorage_size"
  c_size :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCLongStorage_set"
  c_set :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CPtrdiff -> CLong -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCLongStorage_get"
  c_get :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CPtrdiff -> IO CLong

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCLongStorage_new"
  c_new :: Ptr CTHCudaState -> IO (Ptr CTHCudaLongStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCLongStorage_newWithSize"
  c_newWithSize :: Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaLongStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCLongStorage_newWithSize1"
  c_newWithSize1 :: Ptr CTHCudaState -> CLong -> IO (Ptr CTHCudaLongStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCLongStorage_newWithSize2"
  c_newWithSize2 :: Ptr CTHCudaState -> CLong -> CLong -> IO (Ptr CTHCudaLongStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCLongStorage_newWithSize3"
  c_newWithSize3 :: Ptr CTHCudaState -> CLong -> CLong -> CLong -> IO (Ptr CTHCudaLongStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCLongStorage_newWithSize4"
  c_newWithSize4 :: Ptr CTHCudaState -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHCudaLongStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCLongStorage_newWithMapping"
  c_newWithMapping :: Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaLongStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCLongStorage_newWithData"
  c_newWithData :: Ptr CTHCudaState -> Ptr CLong -> CPtrdiff -> IO (Ptr CTHCudaLongStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCLongStorage_setFlag"
  c_setFlag :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCLongStorage_clearFlag"
  c_clearFlag :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCLongStorage_retain"
  c_retain :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCLongStorage_free"
  c_free :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCLongStorage_resize"
  c_resize :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCLongStorage_fill"
  c_fill :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CLong -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCLongStorage_getDevice"
  c_getDevice :: Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCLongStorage_data"
  p_data :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO (Ptr CLong))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCLongStorage_size"
  p_size :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCLongStorage_set"
  p_set :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CPtrdiff -> CLong -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCLongStorage_get"
  p_get :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CPtrdiff -> IO CLong)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCLongStorage_new"
  p_new :: FunPtr (Ptr CTHCudaState -> IO (Ptr CTHCudaLongStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCLongStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaLongStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCLongStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr CTHCudaState -> CLong -> IO (Ptr CTHCudaLongStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCLongStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr CTHCudaState -> CLong -> CLong -> IO (Ptr CTHCudaLongStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCLongStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr CTHCudaState -> CLong -> CLong -> CLong -> IO (Ptr CTHCudaLongStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCLongStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr CTHCudaState -> CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHCudaLongStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCLongStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaLongStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCLongStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CTHCudaState -> Ptr CLong -> CPtrdiff -> IO (Ptr CTHCudaLongStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCLongStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCLongStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCLongStorage_retain"
  p_retain :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCLongStorage_free"
  p_free :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCLongStorage_resize"
  p_resize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCLongStorage_fill"
  p_fill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> CLong -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCLongStorage_getDevice"
  p_getDevice :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongStorage -> IO CInt)