{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.Storage
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
foreign import ccall "THCStorage.h THByteStorage_data"
  c_data :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (Ptr (CUChar))

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THByteStorage_size"
  c_size :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (CPtrdiff)

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THByteStorage_set"
  c_set :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> CPtrdiff -> CUChar -> IO (())

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THByteStorage_get"
  c_get :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> CPtrdiff -> IO (CUChar)

-- | c_new :  state -> THStorage *
foreign import ccall "THCStorage.h THByteStorage_new"
  c_new :: Ptr (CTHState) -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize :  state size -> THStorage *
foreign import ccall "THCStorage.h THByteStorage_newWithSize"
  c_newWithSize :: Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize1 :  state  -> THStorage *
foreign import ccall "THCStorage.h THByteStorage_newWithSize1"
  c_newWithSize1 :: Ptr (CTHState) -> CUChar -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize2 :  state   -> THStorage *
foreign import ccall "THCStorage.h THByteStorage_newWithSize2"
  c_newWithSize2 :: Ptr (CTHState) -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize3 :  state    -> THStorage *
foreign import ccall "THCStorage.h THByteStorage_newWithSize3"
  c_newWithSize3 :: Ptr (CTHState) -> CUChar -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize4 :  state     -> THStorage *
foreign import ccall "THCStorage.h THByteStorage_newWithSize4"
  c_newWithSize4 :: Ptr (CTHState) -> CUChar -> CUChar -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage))

-- | c_newWithMapping :  state filename size shared -> THStorage *
foreign import ccall "THCStorage.h THByteStorage_newWithMapping"
  c_newWithMapping :: Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHByteStorage))

-- | c_newWithData :  state data size -> THStorage *
foreign import ccall "THCStorage.h THByteStorage_newWithData"
  c_newWithData :: Ptr (CTHState) -> Ptr (CUChar) -> CPtrdiff -> IO (Ptr (CTHByteStorage))

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THByteStorage_setFlag"
  c_setFlag :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> CChar -> IO (())

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THByteStorage_clearFlag"
  c_clearFlag :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> CChar -> IO (())

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THByteStorage_retain"
  c_retain :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (())

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THByteStorage_free"
  c_free :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (())

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THByteStorage_resize"
  c_resize :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> CPtrdiff -> IO (())

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THByteStorage_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> CUChar -> IO (())

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THByteStorage_getDevice"
  c_getDevice :: Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (CInt)

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THByteStorage_data"
  p_data :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (Ptr (CUChar)))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THByteStorage_size"
  p_size :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THByteStorage_set"
  p_set :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> CPtrdiff -> CUChar -> IO (()))

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THByteStorage_get"
  p_get :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> CPtrdiff -> IO (CUChar))

-- | p_new : Pointer to function : state -> THStorage *
foreign import ccall "THCStorage.h &THByteStorage_new"
  p_new :: FunPtr (Ptr (CTHState) -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize : Pointer to function : state size -> THStorage *
foreign import ccall "THCStorage.h &THByteStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize1 : Pointer to function : state  -> THStorage *
foreign import ccall "THCStorage.h &THByteStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr (CTHState) -> CUChar -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize2 : Pointer to function : state   -> THStorage *
foreign import ccall "THCStorage.h &THByteStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr (CTHState) -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize3 : Pointer to function : state    -> THStorage *
foreign import ccall "THCStorage.h &THByteStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr (CTHState) -> CUChar -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize4 : Pointer to function : state     -> THStorage *
foreign import ccall "THCStorage.h &THByteStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr (CTHState) -> CUChar -> CUChar -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage)))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THStorage *
foreign import ccall "THCStorage.h &THByteStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHByteStorage)))

-- | p_newWithData : Pointer to function : state data size -> THStorage *
foreign import ccall "THCStorage.h &THByteStorage_newWithData"
  p_newWithData :: FunPtr (Ptr (CTHState) -> Ptr (CUChar) -> CPtrdiff -> IO (Ptr (CTHByteStorage)))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THByteStorage_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THByteStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THByteStorage_retain"
  p_retain :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THByteStorage_free"
  p_free :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THByteStorage_resize"
  p_resize :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THByteStorage_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> CUChar -> IO (()))

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THByteStorage_getDevice"
  p_getDevice :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteStorage) -> IO (CInt))