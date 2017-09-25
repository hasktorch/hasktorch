{-# LANGUAGE ForeignFunctionInterface #-}

module THIntStorage (
    c_THIntStorage_data,
    c_THIntStorage_size,
    c_THIntStorage_elementSize,
    c_THIntStorage_set,
    c_THIntStorage_get,
    c_THIntStorage_new,
    c_THIntStorage_newWithSize,
    c_THIntStorage_newWithSize1,
    c_THIntStorage_newWithSize2,
    c_THIntStorage_newWithSize3,
    c_THIntStorage_newWithSize4,
    c_THIntStorage_newWithMapping,
    c_THIntStorage_newWithData,
    c_THIntStorage_newWithAllocator,
    c_THIntStorage_newWithDataAndAllocator,
    c_THIntStorage_setFlag,
    c_THIntStorage_clearFlag,
    c_THIntStorage_retain,
    c_THIntStorage_swap,
    c_THIntStorage_free,
    c_THIntStorage_resize,
    c_THIntStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntStorage_data :  -> real *
foreign import ccall "THStorage.h THIntStorage_data"
  c_THIntStorage_data :: Ptr CTHIntStorage -> IO (Ptr CInt)

-- |c_THIntStorage_size :  -> ptrdiff_t
foreign import ccall "THStorage.h THIntStorage_size"
  c_THIntStorage_size :: Ptr CTHIntStorage -> CPtrdiff

-- |c_THIntStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THIntStorage_elementSize"
  c_THIntStorage_elementSize :: CSize

-- |c_THIntStorage_set :    -> void
foreign import ccall "THStorage.h THIntStorage_set"
  c_THIntStorage_set :: Ptr CTHIntStorage -> CPtrdiff -> CInt -> IO ()

-- |c_THIntStorage_get :   -> real
foreign import ccall "THStorage.h THIntStorage_get"
  c_THIntStorage_get :: Ptr CTHIntStorage -> CPtrdiff -> CInt

-- |c_THIntStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THIntStorage_new"
  c_THIntStorage_new :: IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize"
  c_THIntStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize1"
  c_THIntStorage_newWithSize1 :: CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize2"
  c_THIntStorage_newWithSize2 :: CInt -> CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize3"
  c_THIntStorage_newWithSize3 :: CInt -> CInt -> CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize4"
  c_THIntStorage_newWithSize4 :: CInt -> CInt -> CInt -> CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithMapping"
  c_THIntStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithData"
  c_THIntStorage_newWithData :: Ptr CInt -> CPtrdiff -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithAllocator"
  c_THIntStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithDataAndAllocator"
  c_THIntStorage_newWithDataAndAllocator :: Ptr CInt -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THIntStorage_setFlag"
  c_THIntStorage_setFlag :: Ptr CTHIntStorage -> CChar -> IO ()

-- |c_THIntStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THIntStorage_clearFlag"
  c_THIntStorage_clearFlag :: Ptr CTHIntStorage -> CChar -> IO ()

-- |c_THIntStorage_retain : storage -> void
foreign import ccall "THStorage.h THIntStorage_retain"
  c_THIntStorage_retain :: Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THIntStorage_swap"
  c_THIntStorage_swap :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_free : storage -> void
foreign import ccall "THStorage.h THIntStorage_free"
  c_THIntStorage_free :: Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_resize : storage size -> void
foreign import ccall "THStorage.h THIntStorage_resize"
  c_THIntStorage_resize :: Ptr CTHIntStorage -> CPtrdiff -> IO ()

-- |c_THIntStorage_fill : storage value -> void
foreign import ccall "THStorage.h THIntStorage_fill"
  c_THIntStorage_fill :: Ptr CTHIntStorage -> CInt -> IO ()