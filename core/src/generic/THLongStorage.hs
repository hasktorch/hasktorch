{-# LANGUAGE ForeignFunctionInterface #-}

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
    c_THLongStorage_fill,
    p_THLongStorage_data,
    p_THLongStorage_size,
    p_THLongStorage_elementSize,
    p_THLongStorage_set,
    p_THLongStorage_get,
    p_THLongStorage_new,
    p_THLongStorage_newWithSize,
    p_THLongStorage_newWithSize1,
    p_THLongStorage_newWithSize2,
    p_THLongStorage_newWithSize3,
    p_THLongStorage_newWithSize4,
    p_THLongStorage_newWithMapping,
    p_THLongStorage_newWithData,
    p_THLongStorage_newWithAllocator,
    p_THLongStorage_newWithDataAndAllocator,
    p_THLongStorage_setFlag,
    p_THLongStorage_clearFlag,
    p_THLongStorage_retain,
    p_THLongStorage_swap,
    p_THLongStorage_free,
    p_THLongStorage_resize,
    p_THLongStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongStorage_data :  -> real *
foreign import ccall unsafe "THStorage.h THLongStorage_data"
  c_THLongStorage_data :: Ptr CTHLongStorage -> IO (Ptr CLong)

-- |c_THLongStorage_size :  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h THLongStorage_size"
  c_THLongStorage_size :: Ptr CTHLongStorage -> CPtrdiff

-- |c_THLongStorage_elementSize :  -> size_t
foreign import ccall unsafe "THStorage.h THLongStorage_elementSize"
  c_THLongStorage_elementSize :: CSize

-- |c_THLongStorage_set :    -> void
foreign import ccall unsafe "THStorage.h THLongStorage_set"
  c_THLongStorage_set :: Ptr CTHLongStorage -> CPtrdiff -> CLong -> IO ()

-- |c_THLongStorage_get :   -> real
foreign import ccall unsafe "THStorage.h THLongStorage_get"
  c_THLongStorage_get :: Ptr CTHLongStorage -> CPtrdiff -> CLong

-- |c_THLongStorage_new :  -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_new"
  c_THLongStorage_new :: IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize : size -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithSize"
  c_THLongStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize1 :  -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithSize1"
  c_THLongStorage_newWithSize1 :: CLong -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize2 :   -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithSize2"
  c_THLongStorage_newWithSize2 :: CLong -> CLong -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize3 :    -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithSize3"
  c_THLongStorage_newWithSize3 :: CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithSize4 :     -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithSize4"
  c_THLongStorage_newWithSize4 :: CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithMapping"
  c_THLongStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithData : data size -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithData"
  c_THLongStorage_newWithData :: Ptr CLong -> CPtrdiff -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithAllocator"
  c_THLongStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h THLongStorage_newWithDataAndAllocator"
  c_THLongStorage_newWithDataAndAllocator :: Ptr CLong -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHLongStorage)

-- |c_THLongStorage_setFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THLongStorage_setFlag"
  c_THLongStorage_setFlag :: Ptr CTHLongStorage -> CChar -> IO ()

-- |c_THLongStorage_clearFlag : storage flag -> void
foreign import ccall unsafe "THStorage.h THLongStorage_clearFlag"
  c_THLongStorage_clearFlag :: Ptr CTHLongStorage -> CChar -> IO ()

-- |c_THLongStorage_retain : storage -> void
foreign import ccall unsafe "THStorage.h THLongStorage_retain"
  c_THLongStorage_retain :: Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_swap : storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h THLongStorage_swap"
  c_THLongStorage_swap :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_free : storage -> void
foreign import ccall unsafe "THStorage.h THLongStorage_free"
  c_THLongStorage_free :: Ptr CTHLongStorage -> IO ()

-- |c_THLongStorage_resize : storage size -> void
foreign import ccall unsafe "THStorage.h THLongStorage_resize"
  c_THLongStorage_resize :: Ptr CTHLongStorage -> CPtrdiff -> IO ()

-- |c_THLongStorage_fill : storage value -> void
foreign import ccall unsafe "THStorage.h THLongStorage_fill"
  c_THLongStorage_fill :: Ptr CTHLongStorage -> CLong -> IO ()

-- |p_THLongStorage_data : Pointer to  -> real *
foreign import ccall unsafe "THStorage.h &THLongStorage_data"
  p_THLongStorage_data :: FunPtr (Ptr CTHLongStorage -> IO (Ptr CLong))

-- |p_THLongStorage_size : Pointer to  -> ptrdiff_t
foreign import ccall unsafe "THStorage.h &THLongStorage_size"
  p_THLongStorage_size :: FunPtr (Ptr CTHLongStorage -> CPtrdiff)

-- |p_THLongStorage_elementSize : Pointer to  -> size_t
foreign import ccall unsafe "THStorage.h &THLongStorage_elementSize"
  p_THLongStorage_elementSize :: FunPtr (CSize)

-- |p_THLongStorage_set : Pointer to    -> void
foreign import ccall unsafe "THStorage.h &THLongStorage_set"
  p_THLongStorage_set :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> CLong -> IO ())

-- |p_THLongStorage_get : Pointer to   -> real
foreign import ccall unsafe "THStorage.h &THLongStorage_get"
  p_THLongStorage_get :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> CLong)

-- |p_THLongStorage_new : Pointer to  -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_new"
  p_THLongStorage_new :: FunPtr (IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithSize : Pointer to size -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithSize"
  p_THLongStorage_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithSize1 : Pointer to  -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithSize1"
  p_THLongStorage_newWithSize1 :: FunPtr (CLong -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithSize2 : Pointer to   -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithSize2"
  p_THLongStorage_newWithSize2 :: FunPtr (CLong -> CLong -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithSize3 : Pointer to    -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithSize3"
  p_THLongStorage_newWithSize3 :: FunPtr (CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithSize4 : Pointer to     -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithSize4"
  p_THLongStorage_newWithSize4 :: FunPtr (CLong -> CLong -> CLong -> CLong -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithMapping : Pointer to filename size flags -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithMapping"
  p_THLongStorage_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithData : Pointer to data size -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithData"
  p_THLongStorage_newWithData :: FunPtr (Ptr CLong -> CPtrdiff -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithAllocator : Pointer to size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithAllocator"
  p_THLongStorage_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_newWithDataAndAllocator : Pointer to data size allocator allocatorContext -> THStorage *
foreign import ccall unsafe "THStorage.h &THLongStorage_newWithDataAndAllocator"
  p_THLongStorage_newWithDataAndAllocator :: FunPtr (Ptr CLong -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHLongStorage))

-- |p_THLongStorage_setFlag : Pointer to storage flag -> void
foreign import ccall unsafe "THStorage.h &THLongStorage_setFlag"
  p_THLongStorage_setFlag :: FunPtr (Ptr CTHLongStorage -> CChar -> IO ())

-- |p_THLongStorage_clearFlag : Pointer to storage flag -> void
foreign import ccall unsafe "THStorage.h &THLongStorage_clearFlag"
  p_THLongStorage_clearFlag :: FunPtr (Ptr CTHLongStorage -> CChar -> IO ())

-- |p_THLongStorage_retain : Pointer to storage -> void
foreign import ccall unsafe "THStorage.h &THLongStorage_retain"
  p_THLongStorage_retain :: FunPtr (Ptr CTHLongStorage -> IO ())

-- |p_THLongStorage_swap : Pointer to storage1 storage2 -> void
foreign import ccall unsafe "THStorage.h &THLongStorage_swap"
  p_THLongStorage_swap :: FunPtr (Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ())

-- |p_THLongStorage_free : Pointer to storage -> void
foreign import ccall unsafe "THStorage.h &THLongStorage_free"
  p_THLongStorage_free :: FunPtr (Ptr CTHLongStorage -> IO ())

-- |p_THLongStorage_resize : Pointer to storage size -> void
foreign import ccall unsafe "THStorage.h &THLongStorage_resize"
  p_THLongStorage_resize :: FunPtr (Ptr CTHLongStorage -> CPtrdiff -> IO ())

-- |p_THLongStorage_fill : Pointer to storage value -> void
foreign import ccall unsafe "THStorage.h &THLongStorage_fill"
  p_THLongStorage_fill :: FunPtr (Ptr CTHLongStorage -> CLong -> IO ())