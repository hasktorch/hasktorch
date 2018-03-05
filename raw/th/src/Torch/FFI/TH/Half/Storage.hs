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
import THTypes
import Data.Word
import Data.Int

-- | c_data :   -> real *
foreign import ccall "THStorage.h c_THStorageHalf_data"
  c_data :: Ptr (CTHHalfStorage) -> IO (Ptr (CTHHalf))

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h c_THStorageHalf_size"
  c_size :: Ptr (CTHHalfStorage) -> IO (CPtrdiff)

-- | c_set :     -> void
foreign import ccall "THStorage.h c_THStorageHalf_set"
  c_set :: Ptr (CTHHalfStorage) -> CPtrdiff -> CTHHalf -> IO (())

-- | c_get :    -> real
foreign import ccall "THStorage.h c_THStorageHalf_get"
  c_get :: Ptr (CTHHalfStorage) -> CPtrdiff -> IO (CTHHalf)

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_new"
  c_new :: IO (Ptr (CTHHalfStorage))

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithSize"
  c_newWithSize :: CPtrdiff -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithSize1"
  c_newWithSize1 :: CTHHalf -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithSize2"
  c_newWithSize2 :: CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithSize3"
  c_newWithSize3 :: CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage))

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithSize4"
  c_newWithSize4 :: CTHHalf -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage))

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithMapping"
  c_newWithMapping :: Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHHalfStorage))

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithData"
  c_newWithData :: Ptr (CTHHalf) -> CPtrdiff -> IO (Ptr (CTHHalfStorage))

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithAllocator"
  c_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHHalfStorage))

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageHalf_newWithDataAndAllocator"
  c_newWithDataAndAllocator :: Ptr (CTHHalf) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHHalfStorage))

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageHalf_setFlag"
  c_setFlag :: Ptr (CTHHalfStorage) -> CChar -> IO (())

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageHalf_clearFlag"
  c_clearFlag :: Ptr (CTHHalfStorage) -> CChar -> IO (())

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h c_THStorageHalf_retain"
  c_retain :: Ptr (CTHHalfStorage) -> IO (())

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h c_THStorageHalf_swap"
  c_swap :: Ptr (CTHHalfStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | c_free :  storage -> void
foreign import ccall "THStorage.h c_THStorageHalf_free"
  c_free :: Ptr (CTHHalfStorage) -> IO (())

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h c_THStorageHalf_resize"
  c_resize :: Ptr (CTHHalfStorage) -> CPtrdiff -> IO (())

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h c_THStorageHalf_fill"
  c_fill :: Ptr (CTHHalfStorage) -> CTHHalf -> IO (())

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &p_THStorageHalf_data"
  p_data :: FunPtr (Ptr (CTHHalfStorage) -> IO (Ptr (CTHHalf)))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &p_THStorageHalf_size"
  p_size :: FunPtr (Ptr (CTHHalfStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &p_THStorageHalf_set"
  p_set :: FunPtr (Ptr (CTHHalfStorage) -> CPtrdiff -> CTHHalf -> IO (()))

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &p_THStorageHalf_get"
  p_get :: FunPtr (Ptr (CTHHalfStorage) -> CPtrdiff -> IO (CTHHalf))

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_new"
  p_new :: FunPtr (IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithSize1"
  p_newWithSize1 :: FunPtr (CTHHalf -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithSize2"
  p_newWithSize2 :: FunPtr (CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithSize3"
  p_newWithSize3 :: FunPtr (CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithSize4"
  p_newWithSize4 :: FunPtr (CTHHalf -> CTHHalf -> CTHHalf -> CTHHalf -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithData"
  p_newWithData :: FunPtr (Ptr (CTHHalf) -> CPtrdiff -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHHalfStorage)))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageHalf_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr (CTHHalf) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHHalfStorage)))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageHalf_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHHalfStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageHalf_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHHalfStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageHalf_retain"
  p_retain :: FunPtr (Ptr (CTHHalfStorage) -> IO (()))

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &p_THStorageHalf_swap"
  p_swap :: FunPtr (Ptr (CTHHalfStorage) -> Ptr (CTHHalfStorage) -> IO (()))

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageHalf_free"
  p_free :: FunPtr (Ptr (CTHHalfStorage) -> IO (()))

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &p_THStorageHalf_resize"
  p_resize :: FunPtr (Ptr (CTHHalfStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &p_THStorageHalf_fill"
  p_fill :: FunPtr (Ptr (CTHHalfStorage) -> CTHHalf -> IO (()))