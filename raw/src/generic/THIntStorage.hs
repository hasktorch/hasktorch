{-# LANGUAGE ForeignFunctionInterface #-}

module THIntStorage (
    c_THIntStorage_data,
    c_THIntStorage_size,
    c_THIntStorage_elementSize,
    c_THIntStorage_set,
    c_THIntStorage_get,
    c_THIntStorage_new,
    c_THIntStorage_newWithSize,
    c_THIntStorage_newWithSize1,
    c_THIntStorage_newWithSize2,
    c_THIntStorage_newWithSize3,
    c_THIntStorage_newWithSize4,
    c_THIntStorage_newWithMapping,
    c_THIntStorage_newWithData,
    c_THIntStorage_newWithAllocator,
    c_THIntStorage_newWithDataAndAllocator,
    c_THIntStorage_setFlag,
    c_THIntStorage_clearFlag,
    c_THIntStorage_retain,
    c_THIntStorage_swap,
    c_THIntStorage_free,
    c_THIntStorage_resize,
    c_THIntStorage_fill,
    p_THIntStorage_data,
    p_THIntStorage_size,
    p_THIntStorage_elementSize,
    p_THIntStorage_set,
    p_THIntStorage_get,
    p_THIntStorage_new,
    p_THIntStorage_newWithSize,
    p_THIntStorage_newWithSize1,
    p_THIntStorage_newWithSize2,
    p_THIntStorage_newWithSize3,
    p_THIntStorage_newWithSize4,
    p_THIntStorage_newWithMapping,
    p_THIntStorage_newWithData,
    p_THIntStorage_newWithAllocator,
    p_THIntStorage_newWithDataAndAllocator,
    p_THIntStorage_setFlag,
    p_THIntStorage_clearFlag,
    p_THIntStorage_retain,
    p_THIntStorage_swap,
    p_THIntStorage_free,
    p_THIntStorage_resize,
    p_THIntStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THIntStorage_data :  -> real *
foreign import ccall "THStorage.h THIntStorage_data"
  c_THIntStorage_data :: Ptr CTHIntStorage -> IO (Ptr CInt)

-- |c_THIntStorage_size :  -> ptrdiff_t
foreign import ccall "THStorage.h THIntStorage_size"
  c_THIntStorage_size :: Ptr CTHIntStorage -> CPtrdiff

-- |c_THIntStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THIntStorage_elementSize"
  c_THIntStorage_elementSize :: CSize

-- |c_THIntStorage_set :    -> void
foreign import ccall "THStorage.h THIntStorage_set"
  c_THIntStorage_set :: Ptr CTHIntStorage -> CPtrdiff -> CInt -> IO ()

-- |c_THIntStorage_get :   -> real
foreign import ccall "THStorage.h THIntStorage_get"
  c_THIntStorage_get :: Ptr CTHIntStorage -> CPtrdiff -> CInt

-- |c_THIntStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THIntStorage_new"
  c_THIntStorage_new :: IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize"
  c_THIntStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize1"
  c_THIntStorage_newWithSize1 :: CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize2"
  c_THIntStorage_newWithSize2 :: CInt -> CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize3"
  c_THIntStorage_newWithSize3 :: CInt -> CInt -> CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithSize4"
  c_THIntStorage_newWithSize4 :: CInt -> CInt -> CInt -> CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithMapping"
  c_THIntStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithData"
  c_THIntStorage_newWithData :: Ptr CInt -> CPtrdiff -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithAllocator"
  c_THIntStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THIntStorage_newWithDataAndAllocator"
  c_THIntStorage_newWithDataAndAllocator :: Ptr CInt -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHIntStorage)

-- |c_THIntStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THIntStorage_setFlag"
  c_THIntStorage_setFlag :: Ptr CTHIntStorage -> CChar -> IO ()

-- |c_THIntStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THIntStorage_clearFlag"
  c_THIntStorage_clearFlag :: Ptr CTHIntStorage -> CChar -> IO ()

-- |c_THIntStorage_retain : storage -> void
foreign import ccall "THStorage.h THIntStorage_retain"
  c_THIntStorage_retain :: Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THIntStorage_swap"
  c_THIntStorage_swap :: Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_free : storage -> void
foreign import ccall "THStorage.h THIntStorage_free"
  c_THIntStorage_free :: Ptr CTHIntStorage -> IO ()

-- |c_THIntStorage_resize : storage size -> void
foreign import ccall "THStorage.h THIntStorage_resize"
  c_THIntStorage_resize :: Ptr CTHIntStorage -> CPtrdiff -> IO ()

-- |c_THIntStorage_fill : storage value -> void
foreign import ccall "THStorage.h THIntStorage_fill"
  c_THIntStorage_fill :: Ptr CTHIntStorage -> CInt -> IO ()

-- |p_THIntStorage_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &THIntStorage_data"
  p_THIntStorage_data :: FunPtr (Ptr CTHIntStorage -> IO (Ptr CInt))

-- |p_THIntStorage_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &THIntStorage_size"
  p_THIntStorage_size :: FunPtr (Ptr CTHIntStorage -> CPtrdiff)

-- |p_THIntStorage_elementSize : Pointer to function :  -> size_t
foreign import ccall "THStorage.h &THIntStorage_elementSize"
  p_THIntStorage_elementSize :: FunPtr (CSize)

-- |p_THIntStorage_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &THIntStorage_set"
  p_THIntStorage_set :: FunPtr (Ptr CTHIntStorage -> CPtrdiff -> CInt -> IO ())

-- |p_THIntStorage_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &THIntStorage_get"
  p_THIntStorage_get :: FunPtr (Ptr CTHIntStorage -> CPtrdiff -> CInt)

-- |p_THIntStorage_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_new"
  p_THIntStorage_new :: FunPtr (IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithSize"
  p_THIntStorage_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithSize1"
  p_THIntStorage_newWithSize1 :: FunPtr (CInt -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithSize2"
  p_THIntStorage_newWithSize2 :: FunPtr (CInt -> CInt -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithSize3"
  p_THIntStorage_newWithSize3 :: FunPtr (CInt -> CInt -> CInt -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithSize4"
  p_THIntStorage_newWithSize4 :: FunPtr (CInt -> CInt -> CInt -> CInt -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithMapping"
  p_THIntStorage_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithData"
  p_THIntStorage_newWithData :: FunPtr (Ptr CInt -> CPtrdiff -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithAllocator"
  p_THIntStorage_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THIntStorage_newWithDataAndAllocator"
  p_THIntStorage_newWithDataAndAllocator :: FunPtr (Ptr CInt -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHIntStorage))

-- |p_THIntStorage_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THIntStorage_setFlag"
  p_THIntStorage_setFlag :: FunPtr (Ptr CTHIntStorage -> CChar -> IO ())

-- |p_THIntStorage_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THIntStorage_clearFlag"
  p_THIntStorage_clearFlag :: FunPtr (Ptr CTHIntStorage -> CChar -> IO ())

-- |p_THIntStorage_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THIntStorage_retain"
  p_THIntStorage_retain :: FunPtr (Ptr CTHIntStorage -> IO ())

-- |p_THIntStorage_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &THIntStorage_swap"
  p_THIntStorage_swap :: FunPtr (Ptr CTHIntStorage -> Ptr CTHIntStorage -> IO ())

-- |p_THIntStorage_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THIntStorage_free"
  p_THIntStorage_free :: FunPtr (Ptr CTHIntStorage -> IO ())

-- |p_THIntStorage_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &THIntStorage_resize"
  p_THIntStorage_resize :: FunPtr (Ptr CTHIntStorage -> CPtrdiff -> IO ())

-- |p_THIntStorage_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &THIntStorage_fill"
  p_THIntStorage_fill :: FunPtr (Ptr CTHIntStorage -> CInt -> IO ())