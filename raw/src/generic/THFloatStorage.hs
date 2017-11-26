{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatStorage (
    c_THFloatStorage_data,
    c_THFloatStorage_size,
    c_THFloatStorage_elementSize,
    c_THFloatStorage_set,
    c_THFloatStorage_get,
    c_THFloatStorage_new,
    c_THFloatStorage_newWithSize,
    c_THFloatStorage_newWithSize1,
    c_THFloatStorage_newWithSize2,
    c_THFloatStorage_newWithSize3,
    c_THFloatStorage_newWithSize4,
    c_THFloatStorage_newWithMapping,
    c_THFloatStorage_newWithData,
    c_THFloatStorage_newWithAllocator,
    c_THFloatStorage_newWithDataAndAllocator,
    c_THFloatStorage_setFlag,
    c_THFloatStorage_clearFlag,
    c_THFloatStorage_retain,
    c_THFloatStorage_swap,
    c_THFloatStorage_free,
    c_THFloatStorage_resize,
    c_THFloatStorage_fill,
    p_THFloatStorage_data,
    p_THFloatStorage_size,
    p_THFloatStorage_elementSize,
    p_THFloatStorage_set,
    p_THFloatStorage_get,
    p_THFloatStorage_new,
    p_THFloatStorage_newWithSize,
    p_THFloatStorage_newWithSize1,
    p_THFloatStorage_newWithSize2,
    p_THFloatStorage_newWithSize3,
    p_THFloatStorage_newWithSize4,
    p_THFloatStorage_newWithMapping,
    p_THFloatStorage_newWithData,
    p_THFloatStorage_newWithAllocator,
    p_THFloatStorage_newWithDataAndAllocator,
    p_THFloatStorage_setFlag,
    p_THFloatStorage_clearFlag,
    p_THFloatStorage_retain,
    p_THFloatStorage_swap,
    p_THFloatStorage_free,
    p_THFloatStorage_resize,
    p_THFloatStorage_fill) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THFloatStorage_data :  -> real *
foreign import ccall "THStorage.h THFloatStorage_data"
  c_THFloatStorage_data :: Ptr CTHFloatStorage -> IO (Ptr CFloat)

-- |c_THFloatStorage_size :  -> ptrdiff_t
foreign import ccall "THStorage.h THFloatStorage_size"
  c_THFloatStorage_size :: Ptr CTHFloatStorage -> CPtrdiff

-- |c_THFloatStorage_elementSize :  -> size_t
foreign import ccall "THStorage.h THFloatStorage_elementSize"
  c_THFloatStorage_elementSize :: CSize

-- |c_THFloatStorage_set :    -> void
foreign import ccall "THStorage.h THFloatStorage_set"
  c_THFloatStorage_set :: Ptr CTHFloatStorage -> CPtrdiff -> CFloat -> IO ()

-- |c_THFloatStorage_get :   -> real
foreign import ccall "THStorage.h THFloatStorage_get"
  c_THFloatStorage_get :: Ptr CTHFloatStorage -> CPtrdiff -> CFloat

-- |c_THFloatStorage_new :  -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_new"
  c_THFloatStorage_new :: IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize"
  c_THFloatStorage_newWithSize :: CPtrdiff -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize1"
  c_THFloatStorage_newWithSize1 :: CFloat -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize2"
  c_THFloatStorage_newWithSize2 :: CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize3"
  c_THFloatStorage_newWithSize3 :: CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithSize4"
  c_THFloatStorage_newWithSize4 :: CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithMapping"
  c_THFloatStorage_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithData"
  c_THFloatStorage_newWithData :: Ptr CFloat -> CPtrdiff -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithAllocator"
  c_THFloatStorage_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h THFloatStorage_newWithDataAndAllocator"
  c_THFloatStorage_newWithDataAndAllocator :: Ptr CFloat -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage)

-- |c_THFloatStorage_setFlag : storage flag -> void
foreign import ccall "THStorage.h THFloatStorage_setFlag"
  c_THFloatStorage_setFlag :: Ptr CTHFloatStorage -> CChar -> IO ()

-- |c_THFloatStorage_clearFlag : storage flag -> void
foreign import ccall "THStorage.h THFloatStorage_clearFlag"
  c_THFloatStorage_clearFlag :: Ptr CTHFloatStorage -> CChar -> IO ()

-- |c_THFloatStorage_retain : storage -> void
foreign import ccall "THStorage.h THFloatStorage_retain"
  c_THFloatStorage_retain :: Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h THFloatStorage_swap"
  c_THFloatStorage_swap :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_free : storage -> void
foreign import ccall "THStorage.h THFloatStorage_free"
  c_THFloatStorage_free :: Ptr CTHFloatStorage -> IO ()

-- |c_THFloatStorage_resize : storage size -> void
foreign import ccall "THStorage.h THFloatStorage_resize"
  c_THFloatStorage_resize :: Ptr CTHFloatStorage -> CPtrdiff -> IO ()

-- |c_THFloatStorage_fill : storage value -> void
foreign import ccall "THStorage.h THFloatStorage_fill"
  c_THFloatStorage_fill :: Ptr CTHFloatStorage -> CFloat -> IO ()

-- |p_THFloatStorage_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &THFloatStorage_data"
  p_THFloatStorage_data :: FunPtr (Ptr CTHFloatStorage -> IO (Ptr CFloat))

-- |p_THFloatStorage_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &THFloatStorage_size"
  p_THFloatStorage_size :: FunPtr (Ptr CTHFloatStorage -> CPtrdiff)

-- |p_THFloatStorage_elementSize : Pointer to function :  -> size_t
foreign import ccall "THStorage.h &THFloatStorage_elementSize"
  p_THFloatStorage_elementSize :: FunPtr (CSize)

-- |p_THFloatStorage_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &THFloatStorage_set"
  p_THFloatStorage_set :: FunPtr (Ptr CTHFloatStorage -> CPtrdiff -> CFloat -> IO ())

-- |p_THFloatStorage_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &THFloatStorage_get"
  p_THFloatStorage_get :: FunPtr (Ptr CTHFloatStorage -> CPtrdiff -> CFloat)

-- |p_THFloatStorage_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_new"
  p_THFloatStorage_new :: FunPtr (IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize"
  p_THFloatStorage_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize1"
  p_THFloatStorage_newWithSize1 :: FunPtr (CFloat -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize2"
  p_THFloatStorage_newWithSize2 :: FunPtr (CFloat -> CFloat -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize3"
  p_THFloatStorage_newWithSize3 :: FunPtr (CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithSize4"
  p_THFloatStorage_newWithSize4 :: FunPtr (CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithMapping"
  p_THFloatStorage_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithData"
  p_THFloatStorage_newWithData :: FunPtr (Ptr CFloat -> CPtrdiff -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithAllocator"
  p_THFloatStorage_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &THFloatStorage_newWithDataAndAllocator"
  p_THFloatStorage_newWithDataAndAllocator :: FunPtr (Ptr CFloat -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage))

-- |p_THFloatStorage_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THFloatStorage_setFlag"
  p_THFloatStorage_setFlag :: FunPtr (Ptr CTHFloatStorage -> CChar -> IO ())

-- |p_THFloatStorage_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &THFloatStorage_clearFlag"
  p_THFloatStorage_clearFlag :: FunPtr (Ptr CTHFloatStorage -> CChar -> IO ())

-- |p_THFloatStorage_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THFloatStorage_retain"
  p_THFloatStorage_retain :: FunPtr (Ptr CTHFloatStorage -> IO ())

-- |p_THFloatStorage_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &THFloatStorage_swap"
  p_THFloatStorage_swap :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_THFloatStorage_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &THFloatStorage_free"
  p_THFloatStorage_free :: FunPtr (Ptr CTHFloatStorage -> IO ())

-- |p_THFloatStorage_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &THFloatStorage_resize"
  p_THFloatStorage_resize :: FunPtr (Ptr CTHFloatStorage -> CPtrdiff -> IO ())

-- |p_THFloatStorage_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &THFloatStorage_fill"
  p_THFloatStorage_fill :: FunPtr (Ptr CTHFloatStorage -> CFloat -> IO ())