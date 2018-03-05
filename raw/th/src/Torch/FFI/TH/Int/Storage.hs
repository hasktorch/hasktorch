{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.Storage
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
foreign import ccall "THStorage.h c_THStorageInt_data"
  c_data :: Ptr (CTHIntStorage) -> IO (Ptr (CInt))

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h c_THStorageInt_size"
  c_size :: Ptr (CTHIntStorage) -> IO (CPtrdiff)

-- | c_set :     -> void
foreign import ccall "THStorage.h c_THStorageInt_set"
  c_set :: Ptr (CTHIntStorage) -> CPtrdiff -> CInt -> IO (())

-- | c_get :    -> real
foreign import ccall "THStorage.h c_THStorageInt_get"
  c_get :: Ptr (CTHIntStorage) -> CPtrdiff -> IO (CInt)

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_new"
  c_new :: IO (Ptr (CTHIntStorage))

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithSize"
  c_newWithSize :: CPtrdiff -> IO (Ptr (CTHIntStorage))

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithSize1"
  c_newWithSize1 :: CInt -> IO (Ptr (CTHIntStorage))

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithSize2"
  c_newWithSize2 :: CInt -> CInt -> IO (Ptr (CTHIntStorage))

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithSize3"
  c_newWithSize3 :: CInt -> CInt -> CInt -> IO (Ptr (CTHIntStorage))

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithSize4"
  c_newWithSize4 :: CInt -> CInt -> CInt -> CInt -> IO (Ptr (CTHIntStorage))

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithMapping"
  c_newWithMapping :: Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHIntStorage))

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithData"
  c_newWithData :: Ptr (CInt) -> CPtrdiff -> IO (Ptr (CTHIntStorage))

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithAllocator"
  c_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHIntStorage))

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageInt_newWithDataAndAllocator"
  c_newWithDataAndAllocator :: Ptr (CInt) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHIntStorage))

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageInt_setFlag"
  c_setFlag :: Ptr (CTHIntStorage) -> CChar -> IO (())

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageInt_clearFlag"
  c_clearFlag :: Ptr (CTHIntStorage) -> CChar -> IO (())

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h c_THStorageInt_retain"
  c_retain :: Ptr (CTHIntStorage) -> IO (())

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h c_THStorageInt_swap"
  c_swap :: Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_free :  storage -> void
foreign import ccall "THStorage.h c_THStorageInt_free"
  c_free :: Ptr (CTHIntStorage) -> IO (())

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h c_THStorageInt_resize"
  c_resize :: Ptr (CTHIntStorage) -> CPtrdiff -> IO (())

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h c_THStorageInt_fill"
  c_fill :: Ptr (CTHIntStorage) -> CInt -> IO (())

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &p_THStorageInt_data"
  p_data :: FunPtr (Ptr (CTHIntStorage) -> IO (Ptr (CInt)))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &p_THStorageInt_size"
  p_size :: FunPtr (Ptr (CTHIntStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &p_THStorageInt_set"
  p_set :: FunPtr (Ptr (CTHIntStorage) -> CPtrdiff -> CInt -> IO (()))

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &p_THStorageInt_get"
  p_get :: FunPtr (Ptr (CTHIntStorage) -> CPtrdiff -> IO (CInt))

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_new"
  p_new :: FunPtr (IO (Ptr (CTHIntStorage)))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr (CTHIntStorage)))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithSize1"
  p_newWithSize1 :: FunPtr (CInt -> IO (Ptr (CTHIntStorage)))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithSize2"
  p_newWithSize2 :: FunPtr (CInt -> CInt -> IO (Ptr (CTHIntStorage)))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithSize3"
  p_newWithSize3 :: FunPtr (CInt -> CInt -> CInt -> IO (Ptr (CTHIntStorage)))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithSize4"
  p_newWithSize4 :: FunPtr (CInt -> CInt -> CInt -> CInt -> IO (Ptr (CTHIntStorage)))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHIntStorage)))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithData"
  p_newWithData :: FunPtr (Ptr (CInt) -> CPtrdiff -> IO (Ptr (CTHIntStorage)))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHIntStorage)))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageInt_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr (CInt) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHIntStorage)))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageInt_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHIntStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageInt_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHIntStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageInt_retain"
  p_retain :: FunPtr (Ptr (CTHIntStorage) -> IO (()))

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &p_THStorageInt_swap"
  p_swap :: FunPtr (Ptr (CTHIntStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageInt_free"
  p_free :: FunPtr (Ptr (CTHIntStorage) -> IO (()))

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &p_THStorageInt_resize"
  p_resize :: FunPtr (Ptr (CTHIntStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &p_THStorageInt_fill"
  p_fill :: FunPtr (Ptr (CTHIntStorage) -> CInt -> IO (()))