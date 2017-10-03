{-# LANGUAGE ForeignFunctionInterface #-}

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
    c_THShortStorage_fill,
    p_THShortStorage_data,
    p_THShortStorage_size,
    p_THShortStorage_elementSize,
    p_THShortStorage_set,
    p_THShortStorage_get,
    p_THShortStorage_new,
    p_THShortStorage_newWithSize,
    p_THShortStorage_newWithSize1,
    p_THShortStorage_newWithSize2,
    p_THShortStorage_newWithSize3,
    p_THShortStorage_newWithSize4,
    p_THShortStorage_newWithMapping,
    p_THShortStorage_newWithData,
    p_THShortStorage_newWithAllocator,
    p_THShortStorage_newWithDataAndAllocator,
    p_THShortStorage_setFlag,
    p_THShortStorage_clearFlag,
    p_THShortStorage_retain,
    p_THShortStorage_swap,
    p_THShortStorage_free,
    p_THShortStorage_resize,
    p_THShortStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortStorage_data :  -> real *
foreign import ccall unsafe "THStorage.h THShortStorage_data"
  c_THShortStorage_data :: Ptr CTHShortStorage -> IO (Ptr CShort)

-- |c_THShortStorage_size :  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h THShortStorage_size"
  c_THShortStorage_size :: Ptr CTHShortStorage -> CPtrdiff

-- |c_THShortStorage_elementSize :  -> size_t
foreign import ccall unsafe "THStorage.h THShortStorage_elementSize"
  c_THShortStorage_elementSize :: CSize

-- |c_THShortStorage_set :    -> void
foreign import ccall unsafe "THStorage.h THShortStorage_set"
  c_THShortStorage_set :: Ptr CTHShortStorage -> CPtrdiff -> CShort -> IO ()

-- |c_THShortStorage_get :   -> real
foreign import ccall unsafe "THStorage.h THShortStorage_get"
  c_THShortStorage_get :: Ptr CTHShortStorage -> CPtrdiff -> CShort

-- |c_THShortStorage_new :  -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_new"
  c_THShortStorage_new :: IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize : size -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithSize"
  c_THShortStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize1 :  -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithSize1"
  c_THShortStorage_newWithSize1 :: CShort -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize2 :   -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithSize2"
  c_THShortStorage_newWithSize2 :: CShort -> CShort -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize3 :    -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithSize3"
  c_THShortStorage_newWithSize3 :: CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithSize4 :     -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithSize4"
  c_THShortStorage_newWithSize4 :: CShort -> CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithMapping"
  c_THShortStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithData : data size -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithData"
  c_THShortStorage_newWithData :: Ptr CShort -> CPtrdiff -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithAllocator"
  c_THShortStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THShortStorage_newWithDataAndAllocator"
  c_THShortStorage_newWithDataAndAllocator :: Ptr CShort -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHShortStorage)

-- |c_THShortStorage_setFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THShortStorage_setFlag"
  c_THShortStorage_setFlag :: Ptr CTHShortStorage -> CChar -> IO ()

-- |c_THShortStorage_clearFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THShortStorage_clearFlag"
  c_THShortStorage_clearFlag :: Ptr CTHShortStorage -> CChar -> IO ()

-- |c_THShortStorage_retain : storage -> void
foreign import ccall unsafe "THStorage.h THShortStorage_retain"
  c_THShortStorage_retain :: Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_swap : storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h THShortStorage_swap"
  c_THShortStorage_swap :: Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_free : storage -> void
foreign import ccall unsafe "THStorage.h THShortStorage_free"
  c_THShortStorage_free :: Ptr CTHShortStorage -> IO ()

-- |c_THShortStorage_resize : storage size -> void
foreign import ccall unsafe "THStorage.h THShortStorage_resize"
  c_THShortStorage_resize :: Ptr CTHShortStorage -> CPtrdiff -> IO ()

-- |c_THShortStorage_fill : storage value -> void
foreign import ccall unsafe "THStorage.h THShortStorage_fill"
  c_THShortStorage_fill :: Ptr CTHShortStorage -> CShort -> IO ()

-- |p_THShortStorage_data : Pointer to function  -> real *
foreign import ccall unsafe "THStorage.h &THShortStorage_data"
  p_THShortStorage_data :: FunPtr (Ptr CTHShortStorage -> IO (Ptr CShort))

-- |p_THShortStorage_size : Pointer to function  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h &THShortStorage_size"
  p_THShortStorage_size :: FunPtr (Ptr CTHShortStorage -> CPtrdiff)

-- |p_THShortStorage_elementSize : Pointer to function  -> size_t
foreign import ccall unsafe "THStorage.h &THShortStorage_elementSize"
  p_THShortStorage_elementSize :: FunPtr (CSize)

-- |p_THShortStorage_set : Pointer to function    -> void
foreign import ccall unsafe "THStorage.h &THShortStorage_set"
  p_THShortStorage_set :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> CShort -> IO ())

-- |p_THShortStorage_get : Pointer to function   -> real
foreign import ccall unsafe "THStorage.h &THShortStorage_get"
  p_THShortStorage_get :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> CShort)

-- |p_THShortStorage_new : Pointer to function  -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_new"
  p_THShortStorage_new :: FunPtr (IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithSize : Pointer to function size -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithSize"
  p_THShortStorage_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithSize1 : Pointer to function  -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithSize1"
  p_THShortStorage_newWithSize1 :: FunPtr (CShort -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithSize2 : Pointer to function   -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithSize2"
  p_THShortStorage_newWithSize2 :: FunPtr (CShort -> CShort -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithSize3 : Pointer to function    -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithSize3"
  p_THShortStorage_newWithSize3 :: FunPtr (CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithSize4 : Pointer to function     -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithSize4"
  p_THShortStorage_newWithSize4 :: FunPtr (CShort -> CShort -> CShort -> CShort -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithMapping : Pointer to function filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithMapping"
  p_THShortStorage_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithData : Pointer to function data size -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithData"
  p_THShortStorage_newWithData :: FunPtr (Ptr CShort -> CPtrdiff -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithAllocator : Pointer to function size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithAllocator"
  p_THShortStorage_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_newWithDataAndAllocator : Pointer to function data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h &THShortStorage_newWithDataAndAllocator"
  p_THShortStorage_newWithDataAndAllocator :: FunPtr (Ptr CShort -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHShortStorage))

-- |p_THShortStorage_setFlag : Pointer to function storage flag -> void
foreign import ccall unsafe "THStorage.h &THShortStorage_setFlag"
  p_THShortStorage_setFlag :: FunPtr (Ptr CTHShortStorage -> CChar -> IO ())

-- |p_THShortStorage_clearFlag : Pointer to function storage flag -> void
foreign import ccall unsafe "THStorage.h &THShortStorage_clearFlag"
  p_THShortStorage_clearFlag :: FunPtr (Ptr CTHShortStorage -> CChar -> IO ())

-- |p_THShortStorage_retain : Pointer to function storage -> void
foreign import ccall unsafe "THStorage.h &THShortStorage_retain"
  p_THShortStorage_retain :: FunPtr (Ptr CTHShortStorage -> IO ())

-- |p_THShortStorage_swap : Pointer to function storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h &THShortStorage_swap"
  p_THShortStorage_swap :: FunPtr (Ptr CTHShortStorage -> Ptr CTHShortStorage -> IO ())

-- |p_THShortStorage_free : Pointer to function storage -> void
foreign import ccall unsafe "THStorage.h &THShortStorage_free"
  p_THShortStorage_free :: FunPtr (Ptr CTHShortStorage -> IO ())

-- |p_THShortStorage_resize : Pointer to function storage size -> void
foreign import ccall unsafe "THStorage.h &THShortStorage_resize"
  p_THShortStorage_resize :: FunPtr (Ptr CTHShortStorage -> CPtrdiff -> IO ())

-- |p_THShortStorage_fill : Pointer to function storage value -> void
foreign import ccall unsafe "THStorage.h &THShortStorage_fill"
  p_THShortStorage_fill :: FunPtr (Ptr CTHShortStorage -> CShort -> IO ())