{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatStorage (
    c_THFloatStorage_data,
    c_THFloatStorage_size,
    c_THFloatStorage_elementSize,
    c_THFloatStorage_set,
    c_THFloatStorage_get,
    c_THFloatStorage_new,
    c_THFloatStorage_newWithSize,
    c_THFloatStorage_newWithSize1,
    c_THFloatStorage_newWithSize2,
    c_THFloatStorage_newWithSize3,
    c_THFloatStorage_newWithSize4,
    c_THFloatStorage_newWithMapping,
    c_THFloatStorage_newWithData,
    c_THFloatStorage_newWithAllocator,
    c_THFloatStorage_newWithDataAndAllocator,
    c_THFloatStorage_setFlag,
    c_THFloatStorage_clearFlag,
    c_THFloatStorage_retain,
    c_THFloatStorage_swap,
    c_THFloatStorage_free,
    c_THFloatStorage_resize,
    c_THFloatStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatStorage_data :  -> real *
foreign import ccall "THStorage.h THFloatStorage_data"
  c_THFloatStorage_data :: Ptr CTHFloatStorage -> IO (Ptr CFloat)

-- |c_THFloatStorage_size :  -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_size"
  c_THFloatStorage_size :: Ptr CTHFloatStorage -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THFloatStorage_elementSize"
  c_THFloatStorage_elementSize :: CSize

-- |c_THFloatStorage_set :    -> void
foreign import ccall "THStorage.h THFloatStorage_set"
  c_THFloatStorage_set :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CFloat -> IO ()

-- |c_THFloatStorage_get :   -> real
foreign import ccall "THStorage.h THFloatStorage_get"
  c_THFloatStorage_get :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> CFloat

-- |c_THFloatStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_new"
  c_THFloatStorage_new :: IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize"
  c_THFloatStorage_newWithSize :: Ptr CTHFloatStorage -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize1"
  c_THFloatStorage_newWithSize1 :: CFloat -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize2"
  c_THFloatStorage_newWithSize2 :: CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize3"
  c_THFloatStorage_newWithSize3 :: CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize4"
  c_THFloatStorage_newWithSize4 :: CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithMapping"
  c_THFloatStorage_newWithMapping :: Ptr CChar -> Ptr CTHFloatStorage -> CInt -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithData"
  c_THFloatStorage_newWithData :: Ptr CFloat -> Ptr CTHFloatStorage -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithAllocator"
  c_THFloatStorage_newWithAllocator :: Ptr CTHFloatStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithDataAndAllocator"
  c_THFloatStorage_newWithDataAndAllocator :: Ptr CFloat -> Ptr CTHFloatStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THFloatStorage_setFlag"
  c_THFloatStorage_setFlag :: Ptr CTHFloatStorage -> CChar -> IO ()

-- |c_THFloatStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THFloatStorage_clearFlag"
  c_THFloatStorage_clearFlag :: Ptr CTHFloatStorage -> CChar -> IO ()

-- |c_THFloatStorage_retain : storage -> void
foreign import ccall "THStorage.h THFloatStorage_retain"
  c_THFloatStorage_retain :: Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THFloatStorage_swap"
  c_THFloatStorage_swap :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_free : storage -> void
foreign import ccall "THStorage.h THFloatStorage_free"
  c_THFloatStorage_free :: Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_resize : storage size -> void
foreign import ccall "THStorage.h THFloatStorage_resize"
  c_THFloatStorage_resize :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_fill : storage value -> void
foreign import ccall "THStorage.h THFloatStorage_fill"
  c_THFloatStorage_fill :: Ptr CTHFloatStorage -> CFloat -> IO ()