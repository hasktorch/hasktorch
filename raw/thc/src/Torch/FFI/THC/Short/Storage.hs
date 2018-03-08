{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.Storage
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
foreign import ccall "THCStorage.h THShortStorage_data"
  c_data :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (Ptr (CShort))

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THShortStorage_size"
  c_size :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (CPtrdiff)

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THShortStorage_set"
  c_set :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> CPtrdiff -> CShort -> IO (())

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THShortStorage_get"
  c_get :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> CPtrdiff -> IO (CShort)

-- | c_new :  state -> THStorage *
foreign import ccall "THCStorage.h THShortStorage_new"
  c_new :: Ptr (CTHState) -> IO (Ptr (CTHShortStorage))

-- | c_newWithSize :  state size -> THStorage *
foreign import ccall "THCStorage.h THShortStorage_newWithSize"
  c_newWithSize :: Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHShortStorage))

-- | c_newWithSize1 :  state  -> THStorage *
foreign import ccall "THCStorage.h THShortStorage_newWithSize1"
  c_newWithSize1 :: Ptr (CTHState) -> CShort -> IO (Ptr (CTHShortStorage))

-- | c_newWithSize2 :  state   -> THStorage *
foreign import ccall "THCStorage.h THShortStorage_newWithSize2"
  c_newWithSize2 :: Ptr (CTHState) -> CShort -> CShort -> IO (Ptr (CTHShortStorage))

-- | c_newWithSize3 :  state    -> THStorage *
foreign import ccall "THCStorage.h THShortStorage_newWithSize3"
  c_newWithSize3 :: Ptr (CTHState) -> CShort -> CShort -> CShort -> IO (Ptr (CTHShortStorage))

-- | c_newWithSize4 :  state     -> THStorage *
foreign import ccall "THCStorage.h THShortStorage_newWithSize4"
  c_newWithSize4 :: Ptr (CTHState) -> CShort -> CShort -> CShort -> CShort -> IO (Ptr (CTHShortStorage))

-- | c_newWithMapping :  state filename size shared -> THStorage *
foreign import ccall "THCStorage.h THShortStorage_newWithMapping"
  c_newWithMapping :: Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHShortStorage))

-- | c_newWithData :  state data size -> THStorage *
foreign import ccall "THCStorage.h THShortStorage_newWithData"
  c_newWithData :: Ptr (CTHState) -> Ptr (CShort) -> CPtrdiff -> IO (Ptr (CTHShortStorage))

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THShortStorage_setFlag"
  c_setFlag :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> CChar -> IO (())

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THShortStorage_clearFlag"
  c_clearFlag :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> CChar -> IO (())

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THShortStorage_retain"
  c_retain :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (())

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THShortStorage_free"
  c_free :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (())

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THShortStorage_resize"
  c_resize :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> CPtrdiff -> IO (())

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THShortStorage_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> CShort -> IO (())

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THShortStorage_getDevice"
  c_getDevice :: Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (CInt)

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THShortStorage_data"
  p_data :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (Ptr (CShort)))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THShortStorage_size"
  p_size :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THShortStorage_set"
  p_set :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> CPtrdiff -> CShort -> IO (()))

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THShortStorage_get"
  p_get :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> CPtrdiff -> IO (CShort))

-- | p_new : Pointer to function : state -> THStorage *
foreign import ccall "THCStorage.h &THShortStorage_new"
  p_new :: FunPtr (Ptr (CTHState) -> IO (Ptr (CTHShortStorage)))

-- | p_newWithSize : Pointer to function : state size -> THStorage *
foreign import ccall "THCStorage.h &THShortStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHShortStorage)))

-- | p_newWithSize1 : Pointer to function : state  -> THStorage *
foreign import ccall "THCStorage.h &THShortStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr (CTHState) -> CShort -> IO (Ptr (CTHShortStorage)))

-- | p_newWithSize2 : Pointer to function : state   -> THStorage *
foreign import ccall "THCStorage.h &THShortStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr (CTHState) -> CShort -> CShort -> IO (Ptr (CTHShortStorage)))

-- | p_newWithSize3 : Pointer to function : state    -> THStorage *
foreign import ccall "THCStorage.h &THShortStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr (CTHState) -> CShort -> CShort -> CShort -> IO (Ptr (CTHShortStorage)))

-- | p_newWithSize4 : Pointer to function : state     -> THStorage *
foreign import ccall "THCStorage.h &THShortStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr (CTHState) -> CShort -> CShort -> CShort -> CShort -> IO (Ptr (CTHShortStorage)))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THStorage *
foreign import ccall "THCStorage.h &THShortStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHShortStorage)))

-- | p_newWithData : Pointer to function : state data size -> THStorage *
foreign import ccall "THCStorage.h &THShortStorage_newWithData"
  p_newWithData :: FunPtr (Ptr (CTHState) -> Ptr (CShort) -> CPtrdiff -> IO (Ptr (CTHShortStorage)))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THShortStorage_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THShortStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THShortStorage_retain"
  p_retain :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THShortStorage_free"
  p_free :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THShortStorage_resize"
  p_resize :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THShortStorage_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> CShort -> IO (()))

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THShortStorage_getDevice"
  p_getDevice :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortStorage) -> IO (CInt))