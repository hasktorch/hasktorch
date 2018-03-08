{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.Storage
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
foreign import ccall "THCStorage.h THFloatStorage_data"
  c_data :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (Ptr (CFloat))

-- | c_size :  state  -> ptrdiff_t
foreign import ccall "THCStorage.h THFloatStorage_size"
  c_size :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (CPtrdiff)

-- | c_set :  state    -> void
foreign import ccall "THCStorage.h THFloatStorage_set"
  c_set :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CPtrdiff -> CFloat -> IO (())

-- | c_get :  state   -> real
foreign import ccall "THCStorage.h THFloatStorage_get"
  c_get :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CPtrdiff -> IO (CFloat)

-- | c_new :  state -> THStorage *
foreign import ccall "THCStorage.h THFloatStorage_new"
  c_new :: Ptr (CTHState) -> IO (Ptr (CTHFloatStorage))

-- | c_newWithSize :  state size -> THStorage *
foreign import ccall "THCStorage.h THFloatStorage_newWithSize"
  c_newWithSize :: Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHFloatStorage))

-- | c_newWithSize1 :  state  -> THStorage *
foreign import ccall "THCStorage.h THFloatStorage_newWithSize1"
  c_newWithSize1 :: Ptr (CTHState) -> CFloat -> IO (Ptr (CTHFloatStorage))

-- | c_newWithSize2 :  state   -> THStorage *
foreign import ccall "THCStorage.h THFloatStorage_newWithSize2"
  c_newWithSize2 :: Ptr (CTHState) -> CFloat -> CFloat -> IO (Ptr (CTHFloatStorage))

-- | c_newWithSize3 :  state    -> THStorage *
foreign import ccall "THCStorage.h THFloatStorage_newWithSize3"
  c_newWithSize3 :: Ptr (CTHState) -> CFloat -> CFloat -> CFloat -> IO (Ptr (CTHFloatStorage))

-- | c_newWithSize4 :  state     -> THStorage *
foreign import ccall "THCStorage.h THFloatStorage_newWithSize4"
  c_newWithSize4 :: Ptr (CTHState) -> CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr (CTHFloatStorage))

-- | c_newWithMapping :  state filename size shared -> THStorage *
foreign import ccall "THCStorage.h THFloatStorage_newWithMapping"
  c_newWithMapping :: Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHFloatStorage))

-- | c_newWithData :  state data size -> THStorage *
foreign import ccall "THCStorage.h THFloatStorage_newWithData"
  c_newWithData :: Ptr (CTHState) -> Ptr (CFloat) -> CPtrdiff -> IO (Ptr (CTHFloatStorage))

-- | c_setFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THFloatStorage_setFlag"
  c_setFlag :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CChar -> IO (())

-- | c_clearFlag :  state storage flag -> void
foreign import ccall "THCStorage.h THFloatStorage_clearFlag"
  c_clearFlag :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CChar -> IO (())

-- | c_retain :  state storage -> void
foreign import ccall "THCStorage.h THFloatStorage_retain"
  c_retain :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_free :  state storage -> void
foreign import ccall "THCStorage.h THFloatStorage_free"
  c_free :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_resize :  state storage size -> void
foreign import ccall "THCStorage.h THFloatStorage_resize"
  c_resize :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CPtrdiff -> IO (())

-- | c_fill :  state storage value -> void
foreign import ccall "THCStorage.h THFloatStorage_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CFloat -> IO (())

-- | c_getDevice :  state storage -> int
foreign import ccall "THCStorage.h THFloatStorage_getDevice"
  c_getDevice :: Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (CInt)

-- | p_data : Pointer to function : state  -> real *
foreign import ccall "THCStorage.h &THFloatStorage_data"
  p_data :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (Ptr (CFloat)))

-- | p_size : Pointer to function : state  -> ptrdiff_t
foreign import ccall "THCStorage.h &THFloatStorage_size"
  p_size :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function : state    -> void
foreign import ccall "THCStorage.h &THFloatStorage_set"
  p_set :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CPtrdiff -> CFloat -> IO (()))

-- | p_get : Pointer to function : state   -> real
foreign import ccall "THCStorage.h &THFloatStorage_get"
  p_get :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CPtrdiff -> IO (CFloat))

-- | p_new : Pointer to function : state -> THStorage *
foreign import ccall "THCStorage.h &THFloatStorage_new"
  p_new :: FunPtr (Ptr (CTHState) -> IO (Ptr (CTHFloatStorage)))

-- | p_newWithSize : Pointer to function : state size -> THStorage *
foreign import ccall "THCStorage.h &THFloatStorage_newWithSize"
  p_newWithSize :: FunPtr (Ptr (CTHState) -> CPtrdiff -> IO (Ptr (CTHFloatStorage)))

-- | p_newWithSize1 : Pointer to function : state  -> THStorage *
foreign import ccall "THCStorage.h &THFloatStorage_newWithSize1"
  p_newWithSize1 :: FunPtr (Ptr (CTHState) -> CFloat -> IO (Ptr (CTHFloatStorage)))

-- | p_newWithSize2 : Pointer to function : state   -> THStorage *
foreign import ccall "THCStorage.h &THFloatStorage_newWithSize2"
  p_newWithSize2 :: FunPtr (Ptr (CTHState) -> CFloat -> CFloat -> IO (Ptr (CTHFloatStorage)))

-- | p_newWithSize3 : Pointer to function : state    -> THStorage *
foreign import ccall "THCStorage.h &THFloatStorage_newWithSize3"
  p_newWithSize3 :: FunPtr (Ptr (CTHState) -> CFloat -> CFloat -> CFloat -> IO (Ptr (CTHFloatStorage)))

-- | p_newWithSize4 : Pointer to function : state     -> THStorage *
foreign import ccall "THCStorage.h &THFloatStorage_newWithSize4"
  p_newWithSize4 :: FunPtr (Ptr (CTHState) -> CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr (CTHFloatStorage)))

-- | p_newWithMapping : Pointer to function : state filename size shared -> THStorage *
foreign import ccall "THCStorage.h &THFloatStorage_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CTHState) -> Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHFloatStorage)))

-- | p_newWithData : Pointer to function : state data size -> THStorage *
foreign import ccall "THCStorage.h &THFloatStorage_newWithData"
  p_newWithData :: FunPtr (Ptr (CTHState) -> Ptr (CFloat) -> CPtrdiff -> IO (Ptr (CTHFloatStorage)))

-- | p_setFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THFloatStorage_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : state storage flag -> void
foreign import ccall "THCStorage.h &THFloatStorage_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THFloatStorage_retain"
  p_retain :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_free : Pointer to function : state storage -> void
foreign import ccall "THCStorage.h &THFloatStorage_free"
  p_free :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_resize : Pointer to function : state storage size -> void
foreign import ccall "THCStorage.h &THFloatStorage_resize"
  p_resize :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : state storage value -> void
foreign import ccall "THCStorage.h &THFloatStorage_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> CFloat -> IO (()))

-- | p_getDevice : Pointer to function : state storage -> int
foreign import ccall "THCStorage.h &THFloatStorage_getDevice"
  p_getDevice :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatStorage) -> IO (CInt))