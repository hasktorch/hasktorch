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
  , c_newWithAllocator
  , c_newWithDataAndAllocator
  , c_setFlag
  , c_clearFlag
  , c_retain
  , c_swap
  , c_free
  , c_resize
  , c_fill
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
  , p_newWithAllocator
  , p_newWithDataAndAllocator
  , p_setFlag
  , p_clearFlag
  , p_retain
  , p_swap
  , p_free
  , p_resize
  , p_fill
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_data :   -> real *
foreign import ccall "THStorage.h c_THStorageByte_data"
  c_data :: Ptr (CTHByteStorage) -> IO (Ptr (CUChar))

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h c_THStorageByte_size"
  c_size :: Ptr (CTHByteStorage) -> IO (CPtrdiff)

-- | c_set :     -> void
foreign import ccall "THStorage.h c_THStorageByte_set"
  c_set :: Ptr (CTHByteStorage) -> CPtrdiff -> CUChar -> IO (())

-- | c_get :    -> real
foreign import ccall "THStorage.h c_THStorageByte_get"
  c_get :: Ptr (CTHByteStorage) -> CPtrdiff -> IO (CUChar)

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_new"
  c_new :: IO (Ptr (CTHByteStorage))

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithSize"
  c_newWithSize :: CPtrdiff -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithSize1"
  c_newWithSize1 :: CUChar -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithSize2"
  c_newWithSize2 :: CUChar -> CUChar -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithSize3"
  c_newWithSize3 :: CUChar -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage))

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithSize4"
  c_newWithSize4 :: CUChar -> CUChar -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage))

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithMapping"
  c_newWithMapping :: Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHByteStorage))

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithData"
  c_newWithData :: Ptr (CUChar) -> CPtrdiff -> IO (Ptr (CTHByteStorage))

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithAllocator"
  c_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHByteStorage))

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageByte_newWithDataAndAllocator"
  c_newWithDataAndAllocator :: Ptr (CUChar) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHByteStorage))

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageByte_setFlag"
  c_setFlag :: Ptr (CTHByteStorage) -> CChar -> IO (())

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageByte_clearFlag"
  c_clearFlag :: Ptr (CTHByteStorage) -> CChar -> IO (())

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h c_THStorageByte_retain"
  c_retain :: Ptr (CTHByteStorage) -> IO (())

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h c_THStorageByte_swap"
  c_swap :: Ptr (CTHByteStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_free :  storage -> void
foreign import ccall "THStorage.h c_THStorageByte_free"
  c_free :: Ptr (CTHByteStorage) -> IO (())

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h c_THStorageByte_resize"
  c_resize :: Ptr (CTHByteStorage) -> CPtrdiff -> IO (())

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h c_THStorageByte_fill"
  c_fill :: Ptr (CTHByteStorage) -> CUChar -> IO (())

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &p_THStorageByte_data"
  p_data :: FunPtr (Ptr (CTHByteStorage) -> IO (Ptr (CUChar)))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &p_THStorageByte_size"
  p_size :: FunPtr (Ptr (CTHByteStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &p_THStorageByte_set"
  p_set :: FunPtr (Ptr (CTHByteStorage) -> CPtrdiff -> CUChar -> IO (()))

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &p_THStorageByte_get"
  p_get :: FunPtr (Ptr (CTHByteStorage) -> CPtrdiff -> IO (CUChar))

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_new"
  p_new :: FunPtr (IO (Ptr (CTHByteStorage)))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithSize1"
  p_newWithSize1 :: FunPtr (CUChar -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithSize2"
  p_newWithSize2 :: FunPtr (CUChar -> CUChar -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithSize3"
  p_newWithSize3 :: FunPtr (CUChar -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage)))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithSize4"
  p_newWithSize4 :: FunPtr (CUChar -> CUChar -> CUChar -> CUChar -> IO (Ptr (CTHByteStorage)))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHByteStorage)))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithData"
  p_newWithData :: FunPtr (Ptr (CUChar) -> CPtrdiff -> IO (Ptr (CTHByteStorage)))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHByteStorage)))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageByte_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr (CUChar) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHByteStorage)))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageByte_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHByteStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageByte_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHByteStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageByte_retain"
  p_retain :: FunPtr (Ptr (CTHByteStorage) -> IO (()))

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &p_THStorageByte_swap"
  p_swap :: FunPtr (Ptr (CTHByteStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageByte_free"
  p_free :: FunPtr (Ptr (CTHByteStorage) -> IO (()))

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &p_THStorageByte_resize"
  p_resize :: FunPtr (Ptr (CTHByteStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &p_THStorageByte_fill"
  p_fill :: FunPtr (Ptr (CTHByteStorage) -> CUChar -> IO (()))