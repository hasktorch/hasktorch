{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.Storage
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
foreign import ccall "THCStorage.h THHalfStorage_data"
  c_data :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (Ptr (CTHHalf))

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THHalfStorage_size"
  c_size :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (CPtrdiff)

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THHalfStorage_set"
  c_set :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CPtrdiff -> CTHHalf -> IO (())

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THHalfStorage_get"
  c_get :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CPtrdiff -> IO (CTHHalf)

-- | c_new :  state -> THStorage *
foreign import ccall "THCStorage.h THHalfStorage_new"
  c_new :: Ptr (CTHState) -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize :  state size -> THStorage *
foreign import ccall "THCStorage.h THHalfStorage_newWithSize"
  c_newWithSize :: Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize1 :  state  -> THStorage *
foreign import ccall "THCStorage.h THHalfStorage_newWithSize1"
  c_newWithSize1 :: Ptr (CTHState) -> CTHHalf -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize2 :  state   -> THStorage *
foreign import ccall "THCStorage.h THHalfStorage_newWithSize2"
  c_newWithSize2 :: Ptr (CTHState) -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize3 :  state    -> THStorage *
foreign import ccall "THCStorage.h THHalfStorage_newWithSize3"
  c_newWithSize3 :: Ptr (CTHState) -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize4 :  state     -> THStorage *
foreign import ccall "THCStorage.h THHalfStorage_newWithSize4"
  c_newWithSize4 :: Ptr (CTHState) -> CTHHalf -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage))

-- | c_newWithMapping :  state filename size shared -> THStorage *
foreign import ccall "THCStorage.h THHalfStorage_newWithMapping"
  c_newWithMapping :: Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHHalfStorage))

-- | c_newWithData :  state data size -> THStorage *
foreign import ccall "THCStorage.h THHalfStorage_newWithData"
  c_newWithData :: Ptr (CTHState) -> Ptr (CTHHalf) -> CPtrdiff -> IO (Ptr (CTHHalfStorage))

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THHalfStorage_setFlag"
  c_setFlag :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CChar -> IO (())

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THHalfStorage_clearFlag"
  c_clearFlag :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CChar -> IO (())

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THHalfStorage_retain"
  c_retain :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (())

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THHalfStorage_free"
  c_free :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (())

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THHalfStorage_resize"
  c_resize :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CPtrdiff -> IO (())

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THHalfStorage_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CTHHalf -> IO (())

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THHalfStorage_getDevice"
  c_getDevice :: Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (CInt)

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THHalfStorage_data"
  p_data :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (Ptr (CTHHalf)))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THHalfStorage_size"
  p_size :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THHalfStorage_set"
  p_set :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CPtrdiff -> CTHHalf -> IO (()))

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THHalfStorage_get"
  p_get :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CPtrdiff -> IO (CTHHalf))

-- | p_new : Pointer to function : state -> THStorage *
foreign import ccall "THCStorage.h &THHalfStorage_new"
  p_new :: FunPtr (Ptr (CTHState) -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize : Pointer to function : state size -> THStorage *
foreign import ccall "THCStorage.h &THHalfStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize1 : Pointer to function : state  -> THStorage *
foreign import ccall "THCStorage.h &THHalfStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr (CTHState) -> CTHHalf -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize2 : Pointer to function : state   -> THStorage *
foreign import ccall "THCStorage.h &THHalfStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr (CTHState) -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize3 : Pointer to function : state    -> THStorage *
foreign import ccall "THCStorage.h &THHalfStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr (CTHState) -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize4 : Pointer to function : state     -> THStorage *
foreign import ccall "THCStorage.h &THHalfStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr (CTHState) -> CTHHalf -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THStorage *
foreign import ccall "THCStorage.h &THHalfStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithData : Pointer to function : state data size -> THStorage *
foreign import ccall "THCStorage.h &THHalfStorage_newWithData"
  p_newWithData :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalf) -> CPtrdiff -> IO (Ptr (CTHHalfStorage)))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THHalfStorage_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THHalfStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THHalfStorage_retain"
  p_retain :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (()))

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THHalfStorage_free"
  p_free :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (()))

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THHalfStorage_resize"
  p_resize :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THHalfStorage_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> CTHHalf -> IO (()))

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THHalfStorage_getDevice"
  p_getDevice :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfStorage) -> IO (CInt))