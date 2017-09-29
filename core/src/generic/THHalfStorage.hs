{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfStorage (
    c_THHalfStorage_data,
    c_THHalfStorage_size,
    c_THHalfStorage_elementSize,
    c_THHalfStorage_set,
    c_THHalfStorage_get,
    c_THHalfStorage_new,
    c_THHalfStorage_newWithSize,
    c_THHalfStorage_newWithSize1,
    c_THHalfStorage_newWithSize2,
    c_THHalfStorage_newWithSize3,
    c_THHalfStorage_newWithSize4,
    c_THHalfStorage_newWithMapping,
    c_THHalfStorage_newWithData,
    c_THHalfStorage_newWithAllocator,
    c_THHalfStorage_newWithDataAndAllocator,
    c_THHalfStorage_setFlag,
    c_THHalfStorage_clearFlag,
    c_THHalfStorage_retain,
    c_THHalfStorage_swap,
    c_THHalfStorage_free,
    c_THHalfStorage_resize,
    c_THHalfStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfStorage_data :  -> real *
foreign import ccall unsafe "THStorage.h THHalfStorage_data"
  c_THHalfStorage_data :: Ptr CTHHalfStorage -> IO (Ptr THHalf)

-- |c_THHalfStorage_size :  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h THHalfStorage_size"
  c_THHalfStorage_size :: Ptr CTHHalfStorage -> CPtrdiff

-- |c_THHalfStorage_elementSize :  -> size_t
foreign import ccall unsafe "THStorage.h THHalfStorage_elementSize"
  c_THHalfStorage_elementSize :: CSize

-- |c_THHalfStorage_set :    -> void
foreign import ccall unsafe "THStorage.h THHalfStorage_set"
  c_THHalfStorage_set :: Ptr CTHHalfStorage -> CPtrdiff -> THHalf -> IO ()

-- |c_THHalfStorage_get :   -> real
foreign import ccall unsafe "THStorage.h THHalfStorage_get"
  c_THHalfStorage_get :: Ptr CTHHalfStorage -> CPtrdiff -> THHalf

-- |c_THHalfStorage_new :  -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_new"
  c_THHalfStorage_new :: IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize : size -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithSize"
  c_THHalfStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize1 :  -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithSize1"
  c_THHalfStorage_newWithSize1 :: THHalf -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize2 :   -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithSize2"
  c_THHalfStorage_newWithSize2 :: THHalf -> THHalf -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize3 :    -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithSize3"
  c_THHalfStorage_newWithSize3 :: THHalf -> THHalf -> THHalf -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize4 :     -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithSize4"
  c_THHalfStorage_newWithSize4 :: THHalf -> THHalf -> THHalf -> THHalf -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithMapping"
  c_THHalfStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithData : data size -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithData"
  c_THHalfStorage_newWithData :: Ptr THHalf -> CPtrdiff -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithAllocator"
  c_THHalfStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THHalfStorage_newWithDataAndAllocator"
  c_THHalfStorage_newWithDataAndAllocator :: Ptr THHalf -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_setFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THHalfStorage_setFlag"
  c_THHalfStorage_setFlag :: Ptr CTHHalfStorage -> CChar -> IO ()

-- |c_THHalfStorage_clearFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THHalfStorage_clearFlag"
  c_THHalfStorage_clearFlag :: Ptr CTHHalfStorage -> CChar -> IO ()

-- |c_THHalfStorage_retain : storage -> void
foreign import ccall unsafe "THStorage.h THHalfStorage_retain"
  c_THHalfStorage_retain :: Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorage_swap : storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h THHalfStorage_swap"
  c_THHalfStorage_swap :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorage_free : storage -> void
foreign import ccall unsafe "THStorage.h THHalfStorage_free"
  c_THHalfStorage_free :: Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorage_resize : storage size -> void
foreign import ccall unsafe "THStorage.h THHalfStorage_resize"
  c_THHalfStorage_resize :: Ptr CTHHalfStorage -> CPtrdiff -> IO ()

-- |c_THHalfStorage_fill : storage value -> void
foreign import ccall unsafe "THStorage.h THHalfStorage_fill"
  c_THHalfStorage_fill :: Ptr CTHHalfStorage -> THHalf -> IO ()