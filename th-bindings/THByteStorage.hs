{-# LANGUAGE ForeignFunctionInterface#-}

module THByteStorage (
    c_THByteStorage_data,
    c_THByteStorage_size,
    c_THByteStorage_elementSize,
    c_THByteStorage_set,
    c_THByteStorage_get,
    c_THByteStorage_new,
    c_THByteStorage_newWithSize,
    c_THByteStorage_newWithSize1,
    c_THByteStorage_newWithSize2,
    c_THByteStorage_newWithSize3,
    c_THByteStorage_newWithSize4,
    c_THByteStorage_newWithMapping,
    c_THByteStorage_newWithData,
    c_THByteStorage_newWithAllocator,
    c_THByteStorage_newWithDataAndAllocator,
    c_THByteStorage_setFlag,
    c_THByteStorage_clearFlag,
    c_THByteStorage_retain,
    c_THByteStorage_swap,
    c_THByteStorage_free,
    c_THByteStorage_resize,
    c_THByteStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteStorage_data :  -> real *
foreign import ccall "THStorage.h THByteStorage_data"
  c_THByteStorage_data :: Ptr CTHByteStorage -> IO (Ptr CChar)

-- |c_THByteStorage_size :  -> THStorage *
foreign import ccall "THStorage.h THByteStorage_size"
  c_THByteStorage_size :: Ptr CTHByteStorage -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THByteStorage_elementSize"
  c_THByteStorage_elementSize :: CSize

-- |c_THByteStorage_set :    -> void
foreign import ccall "THStorage.h THByteStorage_set"
  c_THByteStorage_set :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> CChar -> IO ()

-- |c_THByteStorage_get :   -> real
foreign import ccall "THStorage.h THByteStorage_get"
  c_THByteStorage_get :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> CChar

-- |c_THByteStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THByteStorage_new"
  c_THByteStorage_new :: IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithSize"
  c_THByteStorage_newWithSize :: Ptr CTHByteStorage -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithSize1"
  c_THByteStorage_newWithSize1 :: CChar -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithSize2"
  c_THByteStorage_newWithSize2 :: CChar -> CChar -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithSize3"
  c_THByteStorage_newWithSize3 :: CChar -> CChar -> CChar -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithSize4"
  c_THByteStorage_newWithSize4 :: CChar -> CChar -> CChar -> CChar -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithMapping"
  c_THByteStorage_newWithMapping :: CChar -> Ptr CTHByteStorage -> CInt -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithData"
  c_THByteStorage_newWithData :: Ptr CChar -> Ptr CTHByteStorage -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithAllocator"
  c_THByteStorage_newWithAllocator :: Ptr CTHByteStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THByteStorage_newWithDataAndAllocator"
  c_THByteStorage_newWithDataAndAllocator :: Ptr CChar -> Ptr CTHByteStorage -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THByteStorage_setFlag"
  c_THByteStorage_setFlag :: Ptr CTHByteStorage -> CChar -> IO ()

-- |c_THByteStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THByteStorage_clearFlag"
  c_THByteStorage_clearFlag :: Ptr CTHByteStorage -> CChar -> IO ()

-- |c_THByteStorage_retain : storage -> void
foreign import ccall "THStorage.h THByteStorage_retain"
  c_THByteStorage_retain :: Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THByteStorage_swap"
  c_THByteStorage_swap :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_free : storage -> void
foreign import ccall "THStorage.h THByteStorage_free"
  c_THByteStorage_free :: Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_resize : storage size -> void
foreign import ccall "THStorage.h THByteStorage_resize"
  c_THByteStorage_resize :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_fill : storage value -> void
foreign import ccall "THStorage.h THByteStorage_fill"
  c_THByteStorage_fill :: Ptr CTHByteStorage -> CChar -> IO ()