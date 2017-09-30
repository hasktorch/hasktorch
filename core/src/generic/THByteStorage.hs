{-# LANGUAGE ForeignFunctionInterface #-}

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
    c_THByteStorage_fill,
    p_THByteStorage_data,
    p_THByteStorage_size,
    p_THByteStorage_elementSize,
    p_THByteStorage_set,
    p_THByteStorage_get,
    p_THByteStorage_new,
    p_THByteStorage_newWithSize,
    p_THByteStorage_newWithSize1,
    p_THByteStorage_newWithSize2,
    p_THByteStorage_newWithSize3,
    p_THByteStorage_newWithSize4,
    p_THByteStorage_newWithMapping,
    p_THByteStorage_newWithData,
    p_THByteStorage_newWithAllocator,
    p_THByteStorage_newWithDataAndAllocator,
    p_THByteStorage_setFlag,
    p_THByteStorage_clearFlag,
    p_THByteStorage_retain,
    p_THByteStorage_swap,
    p_THByteStorage_free,
    p_THByteStorage_resize,
    p_THByteStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteStorage_data :  -> real *
foreign import ccall unsafe "THStorage.h THByteStorage_data"
  c_THByteStorage_data :: Ptr CTHByteStorage -> IO (Ptr CChar)

-- |c_THByteStorage_size :  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h THByteStorage_size"
  c_THByteStorage_size :: Ptr CTHByteStorage -> CPtrdiff

-- |c_THByteStorage_elementSize :  -> size_t
foreign import ccall unsafe "THStorage.h THByteStorage_elementSize"
  c_THByteStorage_elementSize :: CSize

-- |c_THByteStorage_set :    -> void
foreign import ccall unsafe "THStorage.h THByteStorage_set"
  c_THByteStorage_set :: Ptr CTHByteStorage -> CPtrdiff -> CChar -> IO ()

-- |c_THByteStorage_get :   -> real
foreign import ccall unsafe "THStorage.h THByteStorage_get"
  c_THByteStorage_get :: Ptr CTHByteStorage -> CPtrdiff -> CChar

-- |c_THByteStorage_new :  -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_new"
  c_THByteStorage_new :: IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize : size -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithSize"
  c_THByteStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize1 :  -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithSize1"
  c_THByteStorage_newWithSize1 :: CChar -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize2 :   -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithSize2"
  c_THByteStorage_newWithSize2 :: CChar -> CChar -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize3 :    -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithSize3"
  c_THByteStorage_newWithSize3 :: CChar -> CChar -> CChar -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithSize4 :     -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithSize4"
  c_THByteStorage_newWithSize4 :: CChar -> CChar -> CChar -> CChar -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithMapping"
  c_THByteStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithData : data size -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithData"
  c_THByteStorage_newWithData :: Ptr CChar -> CPtrdiff -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithAllocator"
  c_THByteStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THByteStorage_newWithDataAndAllocator"
  c_THByteStorage_newWithDataAndAllocator :: Ptr CChar -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHByteStorage)

-- |c_THByteStorage_setFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THByteStorage_setFlag"
  c_THByteStorage_setFlag :: Ptr CTHByteStorage -> CChar -> IO ()

-- |c_THByteStorage_clearFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THByteStorage_clearFlag"
  c_THByteStorage_clearFlag :: Ptr CTHByteStorage -> CChar -> IO ()

-- |c_THByteStorage_retain : storage -> void
foreign import ccall unsafe "THStorage.h THByteStorage_retain"
  c_THByteStorage_retain :: Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_swap : storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h THByteStorage_swap"
  c_THByteStorage_swap :: Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_free : storage -> void
foreign import ccall unsafe "THStorage.h THByteStorage_free"
  c_THByteStorage_free :: Ptr CTHByteStorage -> IO ()

-- |c_THByteStorage_resize : storage size -> void
foreign import ccall unsafe "THStorage.h THByteStorage_resize"
  c_THByteStorage_resize :: Ptr CTHByteStorage -> CPtrdiff -> IO ()

-- |c_THByteStorage_fill : storage value -> void
foreign import ccall unsafe "THStorage.h THByteStorage_fill"
  c_THByteStorage_fill :: Ptr CTHByteStorage -> CChar -> IO ()

-- |p_THByteStorage_data : Pointer to  -> real *
foreign import ccall unsafe "THStorage.h &THByteStorage_data"
  p_THByteStorage_data :: FunPtr (Ptr CTHByteStorage -> IO (Ptr CChar))

-- |p_THByteStorage_size : Pointer to  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h &THByteStorage_size"
  p_THByteStorage_size :: FunPtr (Ptr CTHByteStorage -> CPtrdiff)

-- |p_THByteStorage_elementSize : Pointer to  -> size_t
foreign import ccall unsafe "THStorage.h &THByteStorage_elementSize"
  p_THByteStorage_elementSize :: FunPtr (CSize)

-- |p_THByteStorage_set : Pointer to    -> void
foreign import ccall unsafe "THStorage.h &THByteStorage_set"
  p_THByteStorage_set :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CChar -> IO ())

-- |p_THByteStorage_get : Pointer to   -> real
foreign import ccall unsafe "THStorage.h &THByteStorage_get"
  p_THByteStorage_get :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> CChar)

-- |p_THByteStorage_new : Pointer to  -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_new"
  p_THByteStorage_new :: FunPtr (IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithSize : Pointer to size -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithSize"
  p_THByteStorage_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithSize1 : Pointer to  -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithSize1"
  p_THByteStorage_newWithSize1 :: FunPtr (CChar -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithSize2 : Pointer to   -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithSize2"
  p_THByteStorage_newWithSize2 :: FunPtr (CChar -> CChar -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithSize3 : Pointer to    -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithSize3"
  p_THByteStorage_newWithSize3 :: FunPtr (CChar -> CChar -> CChar -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithSize4 : Pointer to     -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithSize4"
  p_THByteStorage_newWithSize4 :: FunPtr (CChar -> CChar -> CChar -> CChar -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithMapping : Pointer to filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithMapping"
  p_THByteStorage_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithData : Pointer to data size -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithData"
  p_THByteStorage_newWithData :: FunPtr (Ptr CChar -> CPtrdiff -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithAllocator : Pointer to size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithAllocator"
  p_THByteStorage_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_newWithDataAndAllocator : Pointer to data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h &THByteStorage_newWithDataAndAllocator"
  p_THByteStorage_newWithDataAndAllocator :: FunPtr (Ptr CChar -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHByteStorage))

-- |p_THByteStorage_setFlag : Pointer to storage flag -> void
foreign import ccall unsafe "THStorage.h &THByteStorage_setFlag"
  p_THByteStorage_setFlag :: FunPtr (Ptr CTHByteStorage -> CChar -> IO ())

-- |p_THByteStorage_clearFlag : Pointer to storage flag -> void
foreign import ccall unsafe "THStorage.h &THByteStorage_clearFlag"
  p_THByteStorage_clearFlag :: FunPtr (Ptr CTHByteStorage -> CChar -> IO ())

-- |p_THByteStorage_retain : Pointer to storage -> void
foreign import ccall unsafe "THStorage.h &THByteStorage_retain"
  p_THByteStorage_retain :: FunPtr (Ptr CTHByteStorage -> IO ())

-- |p_THByteStorage_swap : Pointer to storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h &THByteStorage_swap"
  p_THByteStorage_swap :: FunPtr (Ptr CTHByteStorage -> Ptr CTHByteStorage -> IO ())

-- |p_THByteStorage_free : Pointer to storage -> void
foreign import ccall unsafe "THStorage.h &THByteStorage_free"
  p_THByteStorage_free :: FunPtr (Ptr CTHByteStorage -> IO ())

-- |p_THByteStorage_resize : Pointer to storage size -> void
foreign import ccall unsafe "THStorage.h &THByteStorage_resize"
  p_THByteStorage_resize :: FunPtr (Ptr CTHByteStorage -> CPtrdiff -> IO ())

-- |p_THByteStorage_fill : Pointer to storage value -> void
foreign import ccall unsafe "THStorage.h &THByteStorage_fill"
  p_THByteStorage_fill :: FunPtr (Ptr CTHByteStorage -> CChar -> IO ())