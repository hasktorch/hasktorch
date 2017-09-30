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
    c_THDoubleStorage_fill,
    p_THDoubleStorage_data,
    p_THDoubleStorage_size,
    p_THDoubleStorage_elementSize,
    p_THDoubleStorage_set,
    p_THDoubleStorage_get,
    p_THDoubleStorage_new,
    p_THDoubleStorage_newWithSize,
    p_THDoubleStorage_newWithSize1,
    p_THDoubleStorage_newWithSize2,
    p_THDoubleStorage_newWithSize3,
    p_THDoubleStorage_newWithSize4,
    p_THDoubleStorage_newWithMapping,
    p_THDoubleStorage_newWithData,
    p_THDoubleStorage_newWithAllocator,
    p_THDoubleStorage_newWithDataAndAllocator,
    p_THDoubleStorage_setFlag,
    p_THDoubleStorage_clearFlag,
    p_THDoubleStorage_retain,
    p_THDoubleStorage_swap,
    p_THDoubleStorage_free,
    p_THDoubleStorage_resize,
    p_THDoubleStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleStorage_data :  -> real *
foreign import ccall unsafe "THStorage.h THDoubleStorage_data"
  c_THDoubleStorage_data :: Ptr CTHDoubleStorage -> IO (Ptr CDouble)

-- |c_THDoubleStorage_size :  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h THDoubleStorage_size"
  c_THDoubleStorage_size :: Ptr CTHDoubleStorage -> CPtrdiff

-- |c_THDoubleStorage_elementSize :  -> size_t
foreign import ccall unsafe "THStorage.h THDoubleStorage_elementSize"
  c_THDoubleStorage_elementSize :: CSize

-- |c_THDoubleStorage_set :    -> void
foreign import ccall unsafe "THStorage.h THDoubleStorage_set"
  c_THDoubleStorage_set :: Ptr CTHDoubleStorage -> CPtrdiff -> CDouble -> IO ()

-- |c_THDoubleStorage_get :   -> real
foreign import ccall unsafe "THStorage.h THDoubleStorage_get"
  c_THDoubleStorage_get :: Ptr CTHDoubleStorage -> CPtrdiff -> CDouble

-- |c_THDoubleStorage_new :  -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_new"
  c_THDoubleStorage_new :: IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize : size -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithSize"
  c_THDoubleStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize1 :  -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithSize1"
  c_THDoubleStorage_newWithSize1 :: CDouble -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize2 :   -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithSize2"
  c_THDoubleStorage_newWithSize2 :: CDouble -> CDouble -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize3 :    -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithSize3"
  c_THDoubleStorage_newWithSize3 :: CDouble -> CDouble -> CDouble -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithSize4 :     -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithSize4"
  c_THDoubleStorage_newWithSize4 :: CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithMapping"
  c_THDoubleStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithData : data size -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithData"
  c_THDoubleStorage_newWithData :: Ptr CDouble -> CPtrdiff -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithAllocator"
  c_THDoubleStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THDoubleStorage_newWithDataAndAllocator"
  c_THDoubleStorage_newWithDataAndAllocator :: Ptr CDouble -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHDoubleStorage)

-- |c_THDoubleStorage_setFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THDoubleStorage_setFlag"
  c_THDoubleStorage_setFlag :: Ptr CTHDoubleStorage -> CChar -> IO ()

-- |c_THDoubleStorage_clearFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THDoubleStorage_clearFlag"
  c_THDoubleStorage_clearFlag :: Ptr CTHDoubleStorage -> CChar -> IO ()

-- |c_THDoubleStorage_retain : storage -> void
foreign import ccall unsafe "THStorage.h THDoubleStorage_retain"
  c_THDoubleStorage_retain :: Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_swap : storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h THDoubleStorage_swap"
  c_THDoubleStorage_swap :: Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_free : storage -> void
foreign import ccall unsafe "THStorage.h THDoubleStorage_free"
  c_THDoubleStorage_free :: Ptr CTHDoubleStorage -> IO ()

-- |c_THDoubleStorage_resize : storage size -> void
foreign import ccall unsafe "THStorage.h THDoubleStorage_resize"
  c_THDoubleStorage_resize :: Ptr CTHDoubleStorage -> CPtrdiff -> IO ()

-- |c_THDoubleStorage_fill : storage value -> void
foreign import ccall unsafe "THStorage.h THDoubleStorage_fill"
  c_THDoubleStorage_fill :: Ptr CTHDoubleStorage -> CDouble -> IO ()

-- |p_THDoubleStorage_data : Pointer to  -> real *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_data"
  p_THDoubleStorage_data :: FunPtr (Ptr CTHDoubleStorage -> IO (Ptr CDouble))

-- |p_THDoubleStorage_size : Pointer to  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h &THDoubleStorage_size"
  p_THDoubleStorage_size :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff)

-- |p_THDoubleStorage_elementSize : Pointer to  -> size_t
foreign import ccall unsafe "THStorage.h &THDoubleStorage_elementSize"
  p_THDoubleStorage_elementSize :: FunPtr (CSize)

-- |p_THDoubleStorage_set : Pointer to    -> void
foreign import ccall unsafe "THStorage.h &THDoubleStorage_set"
  p_THDoubleStorage_set :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff -> CDouble -> IO ())

-- |p_THDoubleStorage_get : Pointer to   -> real
foreign import ccall unsafe "THStorage.h &THDoubleStorage_get"
  p_THDoubleStorage_get :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff -> CDouble)

-- |p_THDoubleStorage_new : Pointer to  -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_new"
  p_THDoubleStorage_new :: FunPtr (IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithSize : Pointer to size -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithSize"
  p_THDoubleStorage_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithSize1 : Pointer to  -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithSize1"
  p_THDoubleStorage_newWithSize1 :: FunPtr (CDouble -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithSize2 : Pointer to   -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithSize2"
  p_THDoubleStorage_newWithSize2 :: FunPtr (CDouble -> CDouble -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithSize3 : Pointer to    -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithSize3"
  p_THDoubleStorage_newWithSize3 :: FunPtr (CDouble -> CDouble -> CDouble -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithSize4 : Pointer to     -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithSize4"
  p_THDoubleStorage_newWithSize4 :: FunPtr (CDouble -> CDouble -> CDouble -> CDouble -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithMapping : Pointer to filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithMapping"
  p_THDoubleStorage_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithData : Pointer to data size -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithData"
  p_THDoubleStorage_newWithData :: FunPtr (Ptr CDouble -> CPtrdiff -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithAllocator : Pointer to size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithAllocator"
  p_THDoubleStorage_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_newWithDataAndAllocator : Pointer to data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h &THDoubleStorage_newWithDataAndAllocator"
  p_THDoubleStorage_newWithDataAndAllocator :: FunPtr (Ptr CDouble -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHDoubleStorage))

-- |p_THDoubleStorage_setFlag : Pointer to storage flag -> void
foreign import ccall unsafe "THStorage.h &THDoubleStorage_setFlag"
  p_THDoubleStorage_setFlag :: FunPtr (Ptr CTHDoubleStorage -> CChar -> IO ())

-- |p_THDoubleStorage_clearFlag : Pointer to storage flag -> void
foreign import ccall unsafe "THStorage.h &THDoubleStorage_clearFlag"
  p_THDoubleStorage_clearFlag :: FunPtr (Ptr CTHDoubleStorage -> CChar -> IO ())

-- |p_THDoubleStorage_retain : Pointer to storage -> void
foreign import ccall unsafe "THStorage.h &THDoubleStorage_retain"
  p_THDoubleStorage_retain :: FunPtr (Ptr CTHDoubleStorage -> IO ())

-- |p_THDoubleStorage_swap : Pointer to storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h &THDoubleStorage_swap"
  p_THDoubleStorage_swap :: FunPtr (Ptr CTHDoubleStorage -> Ptr CTHDoubleStorage -> IO ())

-- |p_THDoubleStorage_free : Pointer to storage -> void
foreign import ccall unsafe "THStorage.h &THDoubleStorage_free"
  p_THDoubleStorage_free :: FunPtr (Ptr CTHDoubleStorage -> IO ())

-- |p_THDoubleStorage_resize : Pointer to storage size -> void
foreign import ccall unsafe "THStorage.h &THDoubleStorage_resize"
  p_THDoubleStorage_resize :: FunPtr (Ptr CTHDoubleStorage -> CPtrdiff -> IO ())

-- |p_THDoubleStorage_fill : Pointer to storage value -> void
foreign import ccall unsafe "THStorage.h &THDoubleStorage_fill"
  p_THDoubleStorage_fill :: FunPtr (Ptr CTHDoubleStorage -> CDouble -> IO ())