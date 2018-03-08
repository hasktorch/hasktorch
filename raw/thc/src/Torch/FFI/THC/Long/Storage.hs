{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.Storage
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
foreign import ccall "THCStorage.h THLongStorage_data"
  c_data :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (Ptr (CLong))

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THLongStorage_size"
  c_size :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (CPtrdiff)

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THLongStorage_set"
  c_set :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> CPtrdiff -> CLong -> IO (())

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THLongStorage_get"
  c_get :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> CPtrdiff -> IO (CLong)

-- | c_new :  state -> THStorage *
foreign import ccall "THCStorage.h THLongStorage_new"
  c_new :: Ptr (CTHState) -> IO (Ptr (CTHLongStorage))

-- | c_newWithSize :  state size -> THStorage *
foreign import ccall "THCStorage.h THLongStorage_newWithSize"
  c_newWithSize :: Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHLongStorage))

-- | c_newWithSize1 :  state  -> THStorage *
foreign import ccall "THCStorage.h THLongStorage_newWithSize1"
  c_newWithSize1 :: Ptr (CTHState) -> CLong -> IO (Ptr (CTHLongStorage))

-- | c_newWithSize2 :  state   -> THStorage *
foreign import ccall "THCStorage.h THLongStorage_newWithSize2"
  c_newWithSize2 :: Ptr (CTHState) -> CLong -> CLong -> IO (Ptr (CTHLongStorage))

-- | c_newWithSize3 :  state    -> THStorage *
foreign import ccall "THCStorage.h THLongStorage_newWithSize3"
  c_newWithSize3 :: Ptr (CTHState) -> CLong -> CLong -> CLong -> IO (Ptr (CTHLongStorage))

-- | c_newWithSize4 :  state     -> THStorage *
foreign import ccall "THCStorage.h THLongStorage_newWithSize4"
  c_newWithSize4 :: Ptr (CTHState) -> CLong -> CLong -> CLong -> CLong -> IO (Ptr (CTHLongStorage))

-- | c_newWithMapping :  state filename size shared -> THStorage *
foreign import ccall "THCStorage.h THLongStorage_newWithMapping"
  c_newWithMapping :: Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHLongStorage))

-- | c_newWithData :  state data size -> THStorage *
foreign import ccall "THCStorage.h THLongStorage_newWithData"
  c_newWithData :: Ptr (CTHState) -> Ptr (CLong) -> CPtrdiff -> IO (Ptr (CTHLongStorage))

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THLongStorage_setFlag"
  c_setFlag :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> CChar -> IO (())

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THLongStorage_clearFlag"
  c_clearFlag :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> CChar -> IO (())

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THLongStorage_retain"
  c_retain :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (())

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THLongStorage_free"
  c_free :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (())

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THLongStorage_resize"
  c_resize :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> CPtrdiff -> IO (())

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THLongStorage_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> CLong -> IO (())

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THLongStorage_getDevice"
  c_getDevice :: Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (CInt)

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THLongStorage_data"
  p_data :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (Ptr (CLong)))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THLongStorage_size"
  p_size :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THLongStorage_set"
  p_set :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> CPtrdiff -> CLong -> IO (()))

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THLongStorage_get"
  p_get :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> CPtrdiff -> IO (CLong))

-- | p_new : Pointer to function : state -> THStorage *
foreign import ccall "THCStorage.h &THLongStorage_new"
  p_new :: FunPtr (Ptr (CTHState) -> IO (Ptr (CTHLongStorage)))

-- | p_newWithSize : Pointer to function : state size -> THStorage *
foreign import ccall "THCStorage.h &THLongStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHLongStorage)))

-- | p_newWithSize1 : Pointer to function : state  -> THStorage *
foreign import ccall "THCStorage.h &THLongStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr (CTHState) -> CLong -> IO (Ptr (CTHLongStorage)))

-- | p_newWithSize2 : Pointer to function : state   -> THStorage *
foreign import ccall "THCStorage.h &THLongStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr (CTHState) -> CLong -> CLong -> IO (Ptr (CTHLongStorage)))

-- | p_newWithSize3 : Pointer to function : state    -> THStorage *
foreign import ccall "THCStorage.h &THLongStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr (CTHState) -> CLong -> CLong -> CLong -> IO (Ptr (CTHLongStorage)))

-- | p_newWithSize4 : Pointer to function : state     -> THStorage *
foreign import ccall "THCStorage.h &THLongStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr (CTHState) -> CLong -> CLong -> CLong -> CLong -> IO (Ptr (CTHLongStorage)))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THStorage *
foreign import ccall "THCStorage.h &THLongStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHLongStorage)))

-- | p_newWithData : Pointer to function : state data size -> THStorage *
foreign import ccall "THCStorage.h &THLongStorage_newWithData"
  p_newWithData :: FunPtr (Ptr (CTHState) -> Ptr (CLong) -> CPtrdiff -> IO (Ptr (CTHLongStorage)))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THLongStorage_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THLongStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THLongStorage_retain"
  p_retain :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THLongStorage_free"
  p_free :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THLongStorage_resize"
  p_resize :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THLongStorage_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> CLong -> IO (()))

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THLongStorage_getDevice"
  p_getDevice :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongStorage) -> IO (CInt))