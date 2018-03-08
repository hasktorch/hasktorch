{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.Storage
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
foreign import ccall "THCStorage.h THDoubleStorage_data"
  c_data :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (Ptr (CDouble))

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THDoubleStorage_size"
  c_size :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (CPtrdiff)

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THDoubleStorage_set"
  c_set :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CPtrdiff -> CDouble -> IO (())

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THDoubleStorage_get"
  c_get :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CPtrdiff -> IO (CDouble)

-- | c_new :  state -> THStorage *
foreign import ccall "THCStorage.h THDoubleStorage_new"
  c_new :: Ptr (CTHState) -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize :  state size -> THStorage *
foreign import ccall "THCStorage.h THDoubleStorage_newWithSize"
  c_newWithSize :: Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize1 :  state  -> THStorage *
foreign import ccall "THCStorage.h THDoubleStorage_newWithSize1"
  c_newWithSize1 :: Ptr (CTHState) -> CDouble -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize2 :  state   -> THStorage *
foreign import ccall "THCStorage.h THDoubleStorage_newWithSize2"
  c_newWithSize2 :: Ptr (CTHState) -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize3 :  state    -> THStorage *
foreign import ccall "THCStorage.h THDoubleStorage_newWithSize3"
  c_newWithSize3 :: Ptr (CTHState) -> CDouble -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize4 :  state     -> THStorage *
foreign import ccall "THCStorage.h THDoubleStorage_newWithSize4"
  c_newWithSize4 :: Ptr (CTHState) -> CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithMapping :  state filename size shared -> THStorage *
foreign import ccall "THCStorage.h THDoubleStorage_newWithMapping"
  c_newWithMapping :: Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithData :  state data size -> THStorage *
foreign import ccall "THCStorage.h THDoubleStorage_newWithData"
  c_newWithData :: Ptr (CTHState) -> Ptr (CDouble) -> CPtrdiff -> IO (Ptr (CTHDoubleStorage))

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THDoubleStorage_setFlag"
  c_setFlag :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CChar -> IO (())

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THDoubleStorage_clearFlag"
  c_clearFlag :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CChar -> IO (())

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THDoubleStorage_retain"
  c_retain :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THDoubleStorage_free"
  c_free :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THDoubleStorage_resize"
  c_resize :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CPtrdiff -> IO (())

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THDoubleStorage_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CDouble -> IO (())

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THDoubleStorage_getDevice"
  c_getDevice :: Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (CInt)

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THDoubleStorage_data"
  p_data :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (Ptr (CDouble)))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THDoubleStorage_size"
  p_size :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THDoubleStorage_set"
  p_set :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CPtrdiff -> CDouble -> IO (()))

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THDoubleStorage_get"
  p_get :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CPtrdiff -> IO (CDouble))

-- | p_new : Pointer to function : state -> THStorage *
foreign import ccall "THCStorage.h &THDoubleStorage_new"
  p_new :: FunPtr (Ptr (CTHState) -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize : Pointer to function : state size -> THStorage *
foreign import ccall "THCStorage.h &THDoubleStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize1 : Pointer to function : state  -> THStorage *
foreign import ccall "THCStorage.h &THDoubleStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr (CTHState) -> CDouble -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize2 : Pointer to function : state   -> THStorage *
foreign import ccall "THCStorage.h &THDoubleStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr (CTHState) -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize3 : Pointer to function : state    -> THStorage *
foreign import ccall "THCStorage.h &THDoubleStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr (CTHState) -> CDouble -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize4 : Pointer to function : state     -> THStorage *
foreign import ccall "THCStorage.h &THDoubleStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr (CTHState) -> CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THStorage *
foreign import ccall "THCStorage.h &THDoubleStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithData : Pointer to function : state data size -> THStorage *
foreign import ccall "THCStorage.h &THDoubleStorage_newWithData"
  p_newWithData :: FunPtr (Ptr (CTHState) -> Ptr (CDouble) -> CPtrdiff -> IO (Ptr (CTHDoubleStorage)))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THDoubleStorage_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THDoubleStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THDoubleStorage_retain"
  p_retain :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THDoubleStorage_free"
  p_free :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THDoubleStorage_resize"
  p_resize :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THDoubleStorage_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> CDouble -> IO (()))

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THDoubleStorage_getDevice"
  p_getDevice :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleStorage) -> IO (CInt))