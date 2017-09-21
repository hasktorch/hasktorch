{-# LANGUAGE ForeignFunctionInterface#-}

module THShortStorage (
    c_THShortStorage_data,
    c_THShortStorage_size,
    c_THShortStorage_elementSize,
    c_THShortStorage_set,
    c_THShortStorage_get,
    c_THShortStorage_new,
    c_THShortStorage_newWithSize,
    c_THShortStorage_newWithSize1,
    c_THShortStorage_newWithSize2,
    c_THShortStorage_newWithSize3,
    c_THShortStorage_newWithSize4,
    c_THShortStorage_newWithMapping,
    c_THShortStorage_newWithData,
    c_THShortStorage_newWithAllocator,
    c_THShortStorage_newWithDataAndAllocator,
    c_THShortStorage_setFlag,
    c_THShortStorage_clearFlag,
    c_THShortStorage_retain,
    c_THShortStorage_swap,
    c_THShortStorage_free,
    c_THShortStorage_resize,
    c_THShortStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortStorage_data :  -> real *
foreign import ccall "THStorage.h THShortStorage_data"
  c_THShortStorage_data :: Ptr CTHShortStorage -> IO (Ptr CShort)

-- |c_THShortStorage_size :  -> THStorage *
foreign import ccall "THStorage.h THShortStorage_size"
  c_THShortStorage_size :: Ptr CTHShortStorage -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THShortStorage_elementSize"
  c_THShortStorage_elementSize :: CSize

-- |c_THShortStorage_set :    -> void
foreign import ccall "THStorage.h THShortStorage_set"
  c_THShortStorage_set :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> CShort -> IO ()

-- |c_THShortStorage_get :   -> real
foreign import ccall "THStorage.h THShortStorage_get"
  c_THShortStorage_get :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> CShort

-- |c_THShortStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THShortStorage_new"
  c_THShortStorage_new :: IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize"
  c_THShortStorage_newWithSize :: Ptr CTHShortStorage -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize1"
  c_THShortStorage_newWithSize1 :: CShort -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize2"
  c_THShortStorage_newWithSize2 :: CShort -> CShort -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize3"
  c_THShortStorage_newWithSize3 :: CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithSize4"
  c_THShortStorage_newWithSize4 :: CShort -> CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithMapping"
  c_THShortStorage_newWithMapping :: CChar -> Ptr CTHShortStorage -> CInt -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithData"
  c_THShortStorage_newWithData :: Ptr CShort -> Ptr CTHShortStorage -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithAllocator"
  c_THShortStorage_newWithAllocator :: Ptr CTHShortStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THShortStorage_newWithDataAndAllocator"
  c_THShortStorage_newWithDataAndAllocator :: Ptr CShort -> Ptr CTHShortStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THShortStorage_setFlag"
  c_THShortStorage_setFlag :: Ptr CTHShortStorage -> CChar -> IO ()

-- |c_THShortStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THShortStorage_clearFlag"
  c_THShortStorage_clearFlag :: Ptr CTHShortStorage -> CChar -> IO ()

-- |c_THShortStorage_retain : storage -> void
foreign import ccall "THStorage.h THShortStorage_retain"
  c_THShortStorage_retain :: Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THShortStorage_swap"
  c_THShortStorage_swap :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_free : storage -> void
foreign import ccall "THStorage.h THShortStorage_free"
  c_THShortStorage_free :: Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_resize : storage size -> void
foreign import ccall "THStorage.h THShortStorage_resize"
  c_THShortStorage_resize :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_fill : storage value -> void
foreign import ccall "THStorage.h THShortStorage_fill"
  c_THShortStorage_fill :: Ptr CTHShortStorage -> CShort -> IO ()