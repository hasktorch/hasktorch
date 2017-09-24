{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleStorage (
    c_THDoubleStorage_data,
    c_THDoubleStorage_size,
    c_THDoubleStorage_elementSize,
    c_THDoubleStorage_set,
    c_THDoubleStorage_get,
    c_THDoubleStorage_new,
    c_THDoubleStorage_newWithSize,
    c_THDoubleStorage_newWithSize1,
    c_THDoubleStorage_newWithSize2,
    c_THDoubleStorage_newWithSize3,
    c_THDoubleStorage_newWithSize4,
    c_THDoubleStorage_newWithMapping,
    c_THDoubleStorage_newWithData,
    c_THDoubleStorage_newWithAllocator,
    c_THDoubleStorage_newWithDataAndAllocator,
    c_THDoubleStorage_setFlag,
    c_THDoubleStorage_clearFlag,
    c_THDoubleStorage_retain,
    c_THDoubleStorage_swap,
    c_THDoubleStorage_free,
    c_THDoubleStorage_resize,
    c_THDoubleStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleStorage_data :  -> real *
foreign import ccall "THStorage.h THDoubleStorage_data"
  c_THDoubleStorage_data :: Ptr CTHDoubleStorage -> IO (Ptr CDouble)

-- |c_THDoubleStorage_size :  -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_size"
  c_THDoubleStorage_size :: Ptr CTHDoubleStorage -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THDoubleStorage_elementSize"
  c_THDoubleStorage_elementSize :: CSize

-- |c_THDoubleStorage_set :    -> void
foreign import ccall "THStorage.h THDoubleStorage_set"
  c_THDoubleStorage_set :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CDouble -> IO ()

-- |c_THDoubleStorage_get :   -> real
foreign import ccall "THStorage.h THDoubleStorage_get"
  c_THDoubleStorage_get :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> CDouble

-- |c_THDoubleStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_new"
  c_THDoubleStorage_new :: IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithSize"
  c_THDoubleStorage_newWithSize :: Ptr CTHDoubleStorage -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithSize1"
  c_THDoubleStorage_newWithSize1 :: CDouble -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithSize2"
  c_THDoubleStorage_newWithSize2 :: CDouble -> CDouble -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithSize3"
  c_THDoubleStorage_newWithSize3 :: CDouble -> CDouble -> CDouble -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithSize4"
  c_THDoubleStorage_newWithSize4 :: CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithMapping"
  c_THDoubleStorage_newWithMapping :: Ptr CChar -> Ptr CTHDoubleStorage -> CInt -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithData"
  c_THDoubleStorage_newWithData :: Ptr CDouble -> Ptr CTHDoubleStorage -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithAllocator"
  c_THDoubleStorage_newWithAllocator :: Ptr CTHDoubleStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THDoubleStorage_newWithDataAndAllocator"
  c_THDoubleStorage_newWithDataAndAllocator :: Ptr CDouble -> Ptr CTHDoubleStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THDoubleStorage_setFlag"
  c_THDoubleStorage_setFlag :: Ptr CTHDoubleStorage -> CChar -> IO ()

-- |c_THDoubleStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THDoubleStorage_clearFlag"
  c_THDoubleStorage_clearFlag :: Ptr CTHDoubleStorage -> CChar -> IO ()

-- |c_THDoubleStorage_retain : storage -> void
foreign import ccall "THStorage.h THDoubleStorage_retain"
  c_THDoubleStorage_retain :: Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THDoubleStorage_swap"
  c_THDoubleStorage_swap :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_free : storage -> void
foreign import ccall "THStorage.h THDoubleStorage_free"
  c_THDoubleStorage_free :: Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_resize : storage size -> void
foreign import ccall "THStorage.h THDoubleStorage_resize"
  c_THDoubleStorage_resize :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_fill : storage value -> void
foreign import ccall "THStorage.h THDoubleStorage_fill"
  c_THDoubleStorage_fill :: Ptr CTHDoubleStorage -> CDouble -> IO ()