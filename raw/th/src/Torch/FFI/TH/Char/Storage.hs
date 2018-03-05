{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.Storage
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
foreign import ccall "THStorage.h c_THStorageChar_data"
  c_data :: Ptr (CTHCharStorage) -> IO (Ptr (CChar))

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h c_THStorageChar_size"
  c_size :: Ptr (CTHCharStorage) -> IO (CPtrdiff)

-- | c_set :     -> void
foreign import ccall "THStorage.h c_THStorageChar_set"
  c_set :: Ptr (CTHCharStorage) -> CPtrdiff -> CChar -> IO (())

-- | c_get :    -> real
foreign import ccall "THStorage.h c_THStorageChar_get"
  c_get :: Ptr (CTHCharStorage) -> CPtrdiff -> IO (CChar)

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_new"
  c_new :: IO (Ptr (CTHCharStorage))

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithSize"
  c_newWithSize :: CPtrdiff -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithSize1"
  c_newWithSize1 :: CChar -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithSize2"
  c_newWithSize2 :: CChar -> CChar -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithSize3"
  c_newWithSize3 :: CChar -> CChar -> CChar -> IO (Ptr (CTHCharStorage))

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithSize4"
  c_newWithSize4 :: CChar -> CChar -> CChar -> CChar -> IO (Ptr (CTHCharStorage))

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithMapping"
  c_newWithMapping :: Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHCharStorage))

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithData"
  c_newWithData :: Ptr (CChar) -> CPtrdiff -> IO (Ptr (CTHCharStorage))

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithAllocator"
  c_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHCharStorage))

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageChar_newWithDataAndAllocator"
  c_newWithDataAndAllocator :: Ptr (CChar) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHCharStorage))

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageChar_setFlag"
  c_setFlag :: Ptr (CTHCharStorage) -> CChar -> IO (())

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageChar_clearFlag"
  c_clearFlag :: Ptr (CTHCharStorage) -> CChar -> IO (())

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h c_THStorageChar_retain"
  c_retain :: Ptr (CTHCharStorage) -> IO (())

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h c_THStorageChar_swap"
  c_swap :: Ptr (CTHCharStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_free :  storage -> void
foreign import ccall "THStorage.h c_THStorageChar_free"
  c_free :: Ptr (CTHCharStorage) -> IO (())

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h c_THStorageChar_resize"
  c_resize :: Ptr (CTHCharStorage) -> CPtrdiff -> IO (())

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h c_THStorageChar_fill"
  c_fill :: Ptr (CTHCharStorage) -> CChar -> IO (())

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &p_THStorageChar_data"
  p_data :: FunPtr (Ptr (CTHCharStorage) -> IO (Ptr (CChar)))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &p_THStorageChar_size"
  p_size :: FunPtr (Ptr (CTHCharStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &p_THStorageChar_set"
  p_set :: FunPtr (Ptr (CTHCharStorage) -> CPtrdiff -> CChar -> IO (()))

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &p_THStorageChar_get"
  p_get :: FunPtr (Ptr (CTHCharStorage) -> CPtrdiff -> IO (CChar))

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_new"
  p_new :: FunPtr (IO (Ptr (CTHCharStorage)))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithSize1"
  p_newWithSize1 :: FunPtr (CChar -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithSize2"
  p_newWithSize2 :: FunPtr (CChar -> CChar -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithSize3"
  p_newWithSize3 :: FunPtr (CChar -> CChar -> CChar -> IO (Ptr (CTHCharStorage)))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithSize4"
  p_newWithSize4 :: FunPtr (CChar -> CChar -> CChar -> CChar -> IO (Ptr (CTHCharStorage)))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHCharStorage)))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithData"
  p_newWithData :: FunPtr (Ptr (CChar) -> CPtrdiff -> IO (Ptr (CTHCharStorage)))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHCharStorage)))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageChar_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr (CChar) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHCharStorage)))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageChar_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHCharStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageChar_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHCharStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageChar_retain"
  p_retain :: FunPtr (Ptr (CTHCharStorage) -> IO (()))

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &p_THStorageChar_swap"
  p_swap :: FunPtr (Ptr (CTHCharStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageChar_free"
  p_free :: FunPtr (Ptr (CTHCharStorage) -> IO (()))

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &p_THStorageChar_resize"
  p_resize :: FunPtr (Ptr (CTHCharStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &p_THStorageChar_fill"
  p_fill :: FunPtr (Ptr (CTHCharStorage) -> CChar -> IO (()))