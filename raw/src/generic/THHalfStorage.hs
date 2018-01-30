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
    c_THHalfStorage_fill,
    p_THHalfStorage_data,
    p_THHalfStorage_size,
    p_THHalfStorage_elementSize,
    p_THHalfStorage_set,
    p_THHalfStorage_get,
    p_THHalfStorage_new,
    p_THHalfStorage_newWithSize,
    p_THHalfStorage_newWithSize1,
    p_THHalfStorage_newWithSize2,
    p_THHalfStorage_newWithSize3,
    p_THHalfStorage_newWithSize4,
    p_THHalfStorage_newWithMapping,
    p_THHalfStorage_newWithData,
    p_THHalfStorage_newWithAllocator,
    p_THHalfStorage_newWithDataAndAllocator,
    p_THHalfStorage_setFlag,
    p_THHalfStorage_clearFlag,
    p_THHalfStorage_retain,
    p_THHalfStorage_swap,
    p_THHalfStorage_free,
    p_THHalfStorage_resize,
    p_THHalfStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THHalfStorage_data :  -> real *
foreign import ccall "THStorage.h THHalfStorage_data"
  c_THHalfStorage_data :: Ptr CTHHalfStorage -> IO (Ptr THHalf)

-- |c_THHalfStorage_size :  -> ptrdiff_t
foreign import ccall "THStorage.h THHalfStorage_size"
  c_THHalfStorage_size :: Ptr CTHHalfStorage -> CPtrdiff

-- |c_THHalfStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THHalfStorage_elementSize"
  c_THHalfStorage_elementSize :: CSize

-- |c_THHalfStorage_set :    -> void
foreign import ccall "THStorage.h THHalfStorage_set"
  c_THHalfStorage_set :: Ptr CTHHalfStorage -> CPtrdiff -> THHalf -> IO ()

-- |c_THHalfStorage_get :   -> real
foreign import ccall "THStorage.h THHalfStorage_get"
  c_THHalfStorage_get :: Ptr CTHHalfStorage -> CPtrdiff -> THHalf

-- |c_THHalfStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_new"
  c_THHalfStorage_new :: IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize"
  c_THHalfStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize1"
  c_THHalfStorage_newWithSize1 :: THHalf -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize2"
  c_THHalfStorage_newWithSize2 :: THHalf -> THHalf -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize3"
  c_THHalfStorage_newWithSize3 :: THHalf -> THHalf -> THHalf -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithSize4"
  c_THHalfStorage_newWithSize4 :: THHalf -> THHalf -> THHalf -> THHalf -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithMapping"
  c_THHalfStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithData"
  c_THHalfStorage_newWithData :: Ptr THHalf -> CPtrdiff -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithAllocator"
  c_THHalfStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THHalfStorage_newWithDataAndAllocator"
  c_THHalfStorage_newWithDataAndAllocator :: Ptr THHalf -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHHalfStorage)

-- |c_THHalfStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THHalfStorage_setFlag"
  c_THHalfStorage_setFlag :: Ptr CTHHalfStorage -> CChar -> IO ()

-- |c_THHalfStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THHalfStorage_clearFlag"
  c_THHalfStorage_clearFlag :: Ptr CTHHalfStorage -> CChar -> IO ()

-- |c_THHalfStorage_retain : storage -> void
foreign import ccall "THStorage.h THHalfStorage_retain"
  c_THHalfStorage_retain :: Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THHalfStorage_swap"
  c_THHalfStorage_swap :: Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorage_free : storage -> void
foreign import ccall "THStorage.h THHalfStorage_free"
  c_THHalfStorage_free :: Ptr CTHHalfStorage -> IO ()

-- |c_THHalfStorage_resize : storage size -> void
foreign import ccall "THStorage.h THHalfStorage_resize"
  c_THHalfStorage_resize :: Ptr CTHHalfStorage -> CPtrdiff -> IO ()

-- |c_THHalfStorage_fill : storage value -> void
foreign import ccall "THStorage.h THHalfStorage_fill"
  c_THHalfStorage_fill :: Ptr CTHHalfStorage -> THHalf -> IO ()

-- |p_THHalfStorage_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &THHalfStorage_data"
  p_THHalfStorage_data :: FunPtr (Ptr CTHHalfStorage -> IO (Ptr THHalf))

-- |p_THHalfStorage_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &THHalfStorage_size"
  p_THHalfStorage_size :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff)

-- |p_THHalfStorage_elementSize : Pointer to function :  -> size_t
foreign import ccall "THStorage.h &THHalfStorage_elementSize"
  p_THHalfStorage_elementSize :: FunPtr (CSize)

-- |p_THHalfStorage_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &THHalfStorage_set"
  p_THHalfStorage_set :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff -> THHalf -> IO ())

-- |p_THHalfStorage_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &THHalfStorage_get"
  p_THHalfStorage_get :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff -> THHalf)

-- |p_THHalfStorage_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_new"
  p_THHalfStorage_new :: FunPtr (IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize"
  p_THHalfStorage_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize1"
  p_THHalfStorage_newWithSize1 :: FunPtr (THHalf -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize2"
  p_THHalfStorage_newWithSize2 :: FunPtr (THHalf -> THHalf -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize3"
  p_THHalfStorage_newWithSize3 :: FunPtr (THHalf -> THHalf -> THHalf -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithSize4"
  p_THHalfStorage_newWithSize4 :: FunPtr (THHalf -> THHalf -> THHalf -> THHalf -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithMapping"
  p_THHalfStorage_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithData"
  p_THHalfStorage_newWithData :: FunPtr (Ptr THHalf -> CPtrdiff -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithAllocator"
  p_THHalfStorage_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THHalfStorage_newWithDataAndAllocator"
  p_THHalfStorage_newWithDataAndAllocator :: FunPtr (Ptr THHalf -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHHalfStorage))

-- |p_THHalfStorage_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THHalfStorage_setFlag"
  p_THHalfStorage_setFlag :: FunPtr (Ptr CTHHalfStorage -> CChar -> IO ())

-- |p_THHalfStorage_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THHalfStorage_clearFlag"
  p_THHalfStorage_clearFlag :: FunPtr (Ptr CTHHalfStorage -> CChar -> IO ())

-- |p_THHalfStorage_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THHalfStorage_retain"
  p_THHalfStorage_retain :: FunPtr (Ptr CTHHalfStorage -> IO ())

-- |p_THHalfStorage_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &THHalfStorage_swap"
  p_THHalfStorage_swap :: FunPtr (Ptr CTHHalfStorage -> Ptr CTHHalfStorage -> IO ())

-- |p_THHalfStorage_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THHalfStorage_free"
  p_THHalfStorage_free :: FunPtr (Ptr CTHHalfStorage -> IO ())

-- |p_THHalfStorage_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &THHalfStorage_resize"
  p_THHalfStorage_resize :: FunPtr (Ptr CTHHalfStorage -> CPtrdiff -> IO ())

-- |p_THHalfStorage_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &THHalfStorage_fill"
  p_THHalfStorage_fill :: FunPtr (Ptr CTHHalfStorage -> THHalf -> IO ())