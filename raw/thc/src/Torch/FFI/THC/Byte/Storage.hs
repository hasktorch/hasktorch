{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Byte.Storage
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
foreign import ccall "THCStorage.h THCByteStorage_data"
  c_data :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO (Ptr CUChar)

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THCByteStorage_size"
  c_size :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO CPtrdiff

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THCByteStorage_set"
  c_set :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CPtrdiff -> CUChar -> IO ()

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THCByteStorage_get"
  c_get :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CPtrdiff -> IO CUChar

-- | c_new :  state -> THCStorage *
foreign import ccall "THCStorage.h THCByteStorage_new"
  c_new :: Ptr CTHCudaState -> IO (Ptr CTHCudaByteStorage)

-- | c_newWithSize :  state size -> THCStorage *
foreign import ccall "THCStorage.h THCByteStorage_newWithSize"
  c_newWithSize :: Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaByteStorage)

-- | c_newWithSize1 :  state  -> THCStorage *
foreign import ccall "THCStorage.h THCByteStorage_newWithSize1"
  c_newWithSize1 :: Ptr CTHCudaState -> CUChar -> IO (Ptr CTHCudaByteStorage)

-- | c_newWithSize2 :  state   -> THCStorage *
foreign import ccall "THCStorage.h THCByteStorage_newWithSize2"
  c_newWithSize2 :: Ptr CTHCudaState -> CUChar -> CUChar -> IO (Ptr CTHCudaByteStorage)

-- | c_newWithSize3 :  state    -> THCStorage *
foreign import ccall "THCStorage.h THCByteStorage_newWithSize3"
  c_newWithSize3 :: Ptr CTHCudaState -> CUChar -> CUChar -> CUChar -> IO (Ptr CTHCudaByteStorage)

-- | c_newWithSize4 :  state     -> THCStorage *
foreign import ccall "THCStorage.h THCByteStorage_newWithSize4"
  c_newWithSize4 :: Ptr CTHCudaState -> CUChar -> CUChar -> CUChar -> CUChar -> IO (Ptr CTHCudaByteStorage)

-- | c_newWithMapping :  state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h THCByteStorage_newWithMapping"
  c_newWithMapping :: Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaByteStorage)

-- | c_newWithData :  state data size -> THCStorage *
foreign import ccall "THCStorage.h THCByteStorage_newWithData"
  c_newWithData :: Ptr CTHCudaState -> Ptr CUChar -> CPtrdiff -> IO (Ptr CTHCudaByteStorage)

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCByteStorage_setFlag"
  c_setFlag :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CChar -> IO ()

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THCByteStorage_clearFlag"
  c_clearFlag :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CChar -> IO ()

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THCByteStorage_retain"
  c_retain :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO ()

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THCByteStorage_free"
  c_free :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO ()

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THCByteStorage_resize"
  c_resize :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CPtrdiff -> IO ()

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THCByteStorage_fill"
  c_fill :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CUChar -> IO ()

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THCByteStorage_getDevice"
  c_getDevice :: Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO CInt

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THCByteStorage_data"
  p_data :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO (Ptr CUChar))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THCByteStorage_size"
  p_size :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO CPtrdiff)

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THCByteStorage_set"
  p_set :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CPtrdiff -> CUChar -> IO ())

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THCByteStorage_get"
  p_get :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CPtrdiff -> IO CUChar)

-- | p_new : Pointer to function : state -> THCStorage *
foreign import ccall "THCStorage.h &THCByteStorage_new"
  p_new :: FunPtr (Ptr CTHCudaState -> IO (Ptr CTHCudaByteStorage))

-- | p_newWithSize : Pointer to function : state size -> THCStorage *
foreign import ccall "THCStorage.h &THCByteStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr CTHCudaState -> CPtrdiff -> IO (Ptr CTHCudaByteStorage))

-- | p_newWithSize1 : Pointer to function : state  -> THCStorage *
foreign import ccall "THCStorage.h &THCByteStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr CTHCudaState -> CUChar -> IO (Ptr CTHCudaByteStorage))

-- | p_newWithSize2 : Pointer to function : state   -> THCStorage *
foreign import ccall "THCStorage.h &THCByteStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr CTHCudaState -> CUChar -> CUChar -> IO (Ptr CTHCudaByteStorage))

-- | p_newWithSize3 : Pointer to function : state    -> THCStorage *
foreign import ccall "THCStorage.h &THCByteStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr CTHCudaState -> CUChar -> CUChar -> CUChar -> IO (Ptr CTHCudaByteStorage))

-- | p_newWithSize4 : Pointer to function : state     -> THCStorage *
foreign import ccall "THCStorage.h &THCByteStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr CTHCudaState -> CUChar -> CUChar -> CUChar -> CUChar -> IO (Ptr CTHCudaByteStorage))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THCStorage *
foreign import ccall "THCStorage.h &THCByteStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CTHCudaState -> Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHCudaByteStorage))

-- | p_newWithData : Pointer to function : state data size -> THCStorage *
foreign import ccall "THCStorage.h &THCByteStorage_newWithData"
  p_newWithData :: FunPtr (Ptr CTHCudaState -> Ptr CUChar -> CPtrdiff -> IO (Ptr CTHCudaByteStorage))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCByteStorage_setFlag"
  p_setFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CChar -> IO ())

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THCByteStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CChar -> IO ())

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCByteStorage_retain"
  p_retain :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO ())

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THCByteStorage_free"
  p_free :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO ())

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THCByteStorage_resize"
  p_resize :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CPtrdiff -> IO ())

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THCByteStorage_fill"
  p_fill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> CUChar -> IO ())

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THCByteStorage_getDevice"
  p_getDevice :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteStorage -> IO CInt)