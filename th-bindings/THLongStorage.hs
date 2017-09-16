{-# LANGUAGE ForeignFunctionInterface#-}

module THLongStorage (
    c_THLongStorage_data,
    c_THLongStorage_size,
    c_THLongStorage_elementSize,
    c_THLongStorage_set,
    c_THLongStorage_get,
    c_THLongStorage_new,
    c_THLongStorage_newWithSize,
    c_THLongStorage_newWithSize1,
    c_THLongStorage_newWithSize2,
    c_THLongStorage_newWithSize3,
    c_THLongStorage_newWithSize4,
    c_THLongStorage_newWithMapping,
    c_THLongStorage_newWithData,
    c_THLongStorage_newWithAllocator,
    c_THLongStorage_newWithDataAndAllocator,
    c_THLongStorage_setFlag,
    c_THLongStorage_clearFlag,
    c_THLongStorage_retain,
    c_THLongStorage_swap,
    c_THLongStorage_free,
    c_THLongStorage_resize,
    c_THLongStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongStorage_data :  -> real *
foreign import ccall "THStorage.h THLongStorage_data"
  c_THLongStorage_data :: Ptr CTHLongStorage -> IO (Ptr CLong)

-- |c_THLongStorage_size :  -> THStorage *
foreign import ccall "THStorage.h THLongStorage_size"
  c_THLongStorage_size :: Ptr CTHLongStorage -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THLongStorage_elementSize"
  c_THLongStorage_elementSize :: CSize

-- |c_THLongStorage_set :    -> void
foreign import ccall "THStorage.h THLongStorage_set"
  c_THLongStorage_set :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> CLong -> IO ()

-- |c_THLongStorage_get :   -> real
foreign import ccall "THStorage.h THLongStorage_get"
  c_THLongStorage_get :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> CLong

-- |c_THLongStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THLongStorage_new"
  c_THLongStorage_new :: IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize"
  c_THLongStorage_newWithSize :: Ptr CTHLongStorage -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize1"
  c_THLongStorage_newWithSize1 :: CLong -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize2"
  c_THLongStorage_newWithSize2 :: CLong -> CLong -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize3"
  c_THLongStorage_newWithSize3 :: CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithSize4"
  c_THLongStorage_newWithSize4 :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithMapping"
  c_THLongStorage_newWithMapping :: CChar -> Ptr CTHLongStorage -> CInt -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithData"
  c_THLongStorage_newWithData :: Ptr CLong -> Ptr CTHLongStorage -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithAllocator"
  c_THLongStorage_newWithAllocator :: Ptr CTHLongStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THLongStorage_newWithDataAndAllocator"
  c_THLongStorage_newWithDataAndAllocator :: Ptr CLong -> Ptr CTHLongStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THLongStorage_setFlag"
  c_THLongStorage_setFlag :: Ptr CTHLongStorage -> CChar -> IO ()

-- |c_THLongStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THLongStorage_clearFlag"
  c_THLongStorage_clearFlag :: Ptr CTHLongStorage -> CChar -> IO ()

-- |c_THLongStorage_retain : storage -> void
foreign import ccall "THStorage.h THLongStorage_retain"
  c_THLongStorage_retain :: Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THLongStorage_swap"
  c_THLongStorage_swap :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_free : storage -> void
foreign import ccall "THStorage.h THLongStorage_free"
  c_THLongStorage_free :: Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_resize : storage size -> void
foreign import ccall "THStorage.h THLongStorage_resize"
  c_THLongStorage_resize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_fill : storage value -> void
foreign import ccall "THStorage.h THLongStorage_fill"
  c_THLongStorage_fill :: Ptr CTHLongStorage -> CLong -> IO ()