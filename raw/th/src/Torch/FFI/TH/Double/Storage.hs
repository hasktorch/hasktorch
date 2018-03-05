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
foreign import ccall "THStorage.h c_THStorageDouble_data"
  c_data :: Ptr (CTHDoubleStorage) -> IO (Ptr (CDouble))

-- | c_size :   -> ptrdiff_t
foreign import ccall "THStorage.h c_THStorageDouble_size"
  c_size :: Ptr (CTHDoubleStorage) -> IO (CPtrdiff)

-- | c_set :     -> void
foreign import ccall "THStorage.h c_THStorageDouble_set"
  c_set :: Ptr (CTHDoubleStorage) -> CPtrdiff -> CDouble -> IO (())

-- | c_get :    -> real
foreign import ccall "THStorage.h c_THStorageDouble_get"
  c_get :: Ptr (CTHDoubleStorage) -> CPtrdiff -> IO (CDouble)

-- | c_new :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_new"
  c_new :: IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize :  size -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithSize"
  c_newWithSize :: CPtrdiff -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize1 :   -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithSize1"
  c_newWithSize1 :: CDouble -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize2 :    -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithSize2"
  c_newWithSize2 :: CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize3 :     -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithSize3"
  c_newWithSize3 :: CDouble -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithSize4 :      -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithSize4"
  c_newWithSize4 :: CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithMapping :  filename size flags -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithMapping"
  c_newWithMapping :: Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithData :  data size -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithData"
  c_newWithData :: Ptr (CDouble) -> CPtrdiff -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithAllocator :  size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithAllocator"
  c_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHDoubleStorage))

-- | c_newWithDataAndAllocator :  data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h c_THStorageDouble_newWithDataAndAllocator"
  c_newWithDataAndAllocator :: Ptr (CDouble) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHDoubleStorage))

-- | c_setFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageDouble_setFlag"
  c_setFlag :: Ptr (CTHDoubleStorage) -> CChar -> IO (())

-- | c_clearFlag :  storage flag -> void
foreign import ccall "THStorage.h c_THStorageDouble_clearFlag"
  c_clearFlag :: Ptr (CTHDoubleStorage) -> CChar -> IO (())

-- | c_retain :  storage -> void
foreign import ccall "THStorage.h c_THStorageDouble_retain"
  c_retain :: Ptr (CTHDoubleStorage) -> IO (())

-- | c_swap :  storage1 storage2 -> void
foreign import ccall "THStorage.h c_THStorageDouble_swap"
  c_swap :: Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_free :  storage -> void
foreign import ccall "THStorage.h c_THStorageDouble_free"
  c_free :: Ptr (CTHDoubleStorage) -> IO (())

-- | c_resize :  storage size -> void
foreign import ccall "THStorage.h c_THStorageDouble_resize"
  c_resize :: Ptr (CTHDoubleStorage) -> CPtrdiff -> IO (())

-- | c_fill :  storage value -> void
foreign import ccall "THStorage.h c_THStorageDouble_fill"
  c_fill :: Ptr (CTHDoubleStorage) -> CDouble -> IO (())

-- | p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &p_THStorageDouble_data"
  p_data :: FunPtr (Ptr (CTHDoubleStorage) -> IO (Ptr (CDouble)))

-- | p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &p_THStorageDouble_size"
  p_size :: FunPtr (Ptr (CTHDoubleStorage) -> IO (CPtrdiff))

-- | p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &p_THStorageDouble_set"
  p_set :: FunPtr (Ptr (CTHDoubleStorage) -> CPtrdiff -> CDouble -> IO (()))

-- | p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &p_THStorageDouble_get"
  p_get :: FunPtr (Ptr (CTHDoubleStorage) -> CPtrdiff -> IO (CDouble))

-- | p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_new"
  p_new :: FunPtr (IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithSize1"
  p_newWithSize1 :: FunPtr (CDouble -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithSize2"
  p_newWithSize2 :: FunPtr (CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithSize3"
  p_newWithSize3 :: FunPtr (CDouble -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithSize4"
  p_newWithSize4 :: FunPtr (CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithMapping"
  p_newWithMapping :: FunPtr (Ptr (CChar) -> CPtrdiff -> CInt -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithData"
  p_newWithData :: FunPtr (Ptr (CDouble) -> CPtrdiff -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHDoubleStorage)))

-- | p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &p_THStorageDouble_newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr (CDouble) -> CPtrdiff -> CTHAllocatorPtr -> Ptr (()) -> IO (Ptr (CTHDoubleStorage)))

-- | p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageDouble_setFlag"
  p_setFlag :: FunPtr (Ptr (CTHDoubleStorage) -> CChar -> IO (()))

-- | p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &p_THStorageDouble_clearFlag"
  p_clearFlag :: FunPtr (Ptr (CTHDoubleStorage) -> CChar -> IO (()))

-- | p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageDouble_retain"
  p_retain :: FunPtr (Ptr (CTHDoubleStorage) -> IO (()))

-- | p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &p_THStorageDouble_swap"
  p_swap :: FunPtr (Ptr (CTHDoubleStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &p_THStorageDouble_free"
  p_free :: FunPtr (Ptr (CTHDoubleStorage) -> IO (()))

-- | p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &p_THStorageDouble_resize"
  p_resize :: FunPtr (Ptr (CTHDoubleStorage) -> CPtrdiff -> IO (()))

-- | p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &p_THStorageDouble_fill"
  p_fill :: FunPtr (Ptr (CTHDoubleStorage) -> CDouble -> IO (()))