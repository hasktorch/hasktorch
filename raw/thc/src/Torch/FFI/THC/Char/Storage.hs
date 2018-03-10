{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.Storage
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
foreign import ccall "THCStorage.h THCCharStorage_data"
  c_data :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO (Ptr CChar)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCCharStorage_size"
  c_size :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCCharStorage_set"
  c_set :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CPtrdiff -> CChar -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCCharStorage_get"
  c_get :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CPtrdiff -> IO CChar

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCCharStorage_new"
  c_new :: Ptr CTHCudaState -> IO (Ptr CTHCudaCharStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCCharStorage_newWithSize"
  c_newWithSize :: Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaCharStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCCharStorage_newWithSize1"
  c_newWithSize1 :: Ptr CTHCudaState -> CChar -> IO (Ptr CTHCudaCharStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCCharStorage_newWithSize2"
  c_newWithSize2 :: Ptr CTHCudaState -> CChar -> CChar -> IO (Ptr CTHCudaCharStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCCharStorage_newWithSize3"
  c_newWithSize3 :: Ptr CTHCudaState -> CChar -> CChar -> CChar -> IO (Ptr CTHCudaCharStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCCharStorage_newWithSize4"
  c_newWithSize4 :: Ptr CTHCudaState -> CChar -> CChar -> CChar -> CChar -> IO (Ptr CTHCudaCharStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCCharStorage_newWithMapping"
  c_newWithMapping :: Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaCharStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCCharStorage_newWithData"
  c_newWithData :: Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> IO (Ptr CTHCudaCharStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCCharStorage_setFlag"
  c_setFlag :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCCharStorage_clearFlag"
  c_clearFlag :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCCharStorage_retain"
  c_retain :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCCharStorage_free"
  c_free :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCCharStorage_resize"
  c_resize :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCCharStorage_fill"
  c_fill :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CChar -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCCharStorage_getDevice"
  c_getDevice :: Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCCharStorage_data"
  p_data :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO (Ptr CChar))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCCharStorage_size"
  p_size :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCCharStorage_set"
  p_set :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CPtrdiff -> CChar -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCCharStorage_get"
  p_get :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CPtrdiff -> IO CChar)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCCharStorage_new"
  p_new :: FunPtr (Ptr CTHCudaState -> IO (Ptr CTHCudaCharStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCCharStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaCharStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCCharStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr CTHCudaState -> CChar -> IO (Ptr CTHCudaCharStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCCharStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr CTHCudaState -> CChar -> CChar -> IO (Ptr CTHCudaCharStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCCharStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr CTHCudaState -> CChar -> CChar -> CChar -> IO (Ptr CTHCudaCharStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCCharStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr CTHCudaState -> CChar -> CChar -> CChar -> CChar -> IO (Ptr CTHCudaCharStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCCharStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaCharStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCCharStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> IO (Ptr CTHCudaCharStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCCharStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCCharStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCCharStorage_retain"
  p_retain :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCCharStorage_free"
  p_free :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCCharStorage_resize"
  p_resize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCCharStorage_fill"
  p_fill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> CChar -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCCharStorage_getDevice"
  p_getDevice :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharStorage -> IO CInt)