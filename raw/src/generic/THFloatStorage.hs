{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatStorage
  ( c_data
  , c_size
  , c_elementSize
  , c_set
  , c_get
  , c_new
  , c_newWithSize
  , c_newWithSize1
  , c_newWithSize2
  , c_newWithSize3
  , c_newWithSize4
  , c_newWithMapping
  , c_newWithData
  , c_newWithAllocator
  , c_newWithDataAndAllocator
  , c_setFlag
  , c_clearFlag
  , c_retain
  , c_swap
  , c_free
  , c_resize
  , c_fill
  , p_data
  , p_size
  , p_elementSize
  , p_set
  , p_get
  , p_new
  , p_newWithSize
  , p_newWithSize1
  , p_newWithSize2
  , p_newWithSize3
  , p_newWithSize4
  , p_newWithMapping
  , p_newWithData
  , p_newWithAllocator
  , p_newWithDataAndAllocator
  , p_setFlag
  , p_clearFlag
  , p_retain
  , p_swap
  , p_free
  , p_resize
  , p_fill
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_data :  -> real *
foreign import ccall "THStorage.h data"
  c_data :: Ptr CTHFloatStorage -> IO (Ptr CFloat)

-- | c_size :  -> ptrdiff_t
foreign import ccall "THStorage.h size"
  c_size :: Ptr CTHFloatStorage -> CPtrdiff

-- | c_elementSize :  -> size_t
foreign import ccall "THStorage.h elementSize"
  c_elementSize :: CSize

-- | c_set :    -> void
foreign import ccall "THStorage.h set"
  c_set :: Ptr CTHFloatStorage -> CPtrdiff -> CFloat -> IO ()

-- | c_get :   -> real
foreign import ccall "THStorage.h get"
  c_get :: Ptr CTHFloatStorage -> CPtrdiff -> CFloat

-- | c_new :  -> THStorage *
foreign import ccall "THStorage.h new"
  c_new :: IO (Ptr CTHFloatStorage)

-- | c_newWithSize : size -> THStorage *
foreign import ccall "THStorage.h newWithSize"
  c_newWithSize :: CPtrdiff -> IO (Ptr CTHFloatStorage)

-- | c_newWithSize1 :  -> THStorage *
foreign import ccall "THStorage.h newWithSize1"
  c_newWithSize1 :: CFloat -> IO (Ptr CTHFloatStorage)

-- | c_newWithSize2 :   -> THStorage *
foreign import ccall "THStorage.h newWithSize2"
  c_newWithSize2 :: CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- | c_newWithSize3 :    -> THStorage *
foreign import ccall "THStorage.h newWithSize3"
  c_newWithSize3 :: CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- | c_newWithSize4 :     -> THStorage *
foreign import ccall "THStorage.h newWithSize4"
  c_newWithSize4 :: CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage)

-- | c_newWithMapping : filename size flags -> THStorage *
foreign import ccall "THStorage.h newWithMapping"
  c_newWithMapping :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHFloatStorage)

-- | c_newWithData : data size -> THStorage *
foreign import ccall "THStorage.h newWithData"
  c_newWithData :: Ptr CFloat -> CPtrdiff -> IO (Ptr CTHFloatStorage)

-- | c_newWithAllocator : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h newWithAllocator"
  c_newWithAllocator :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage)

-- | c_newWithDataAndAllocator : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h newWithDataAndAllocator"
  c_newWithDataAndAllocator :: Ptr CFloat -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage)

-- | c_setFlag : storage flag -> void
foreign import ccall "THStorage.h setFlag"
  c_setFlag :: Ptr CTHFloatStorage -> CChar -> IO ()

-- | c_clearFlag : storage flag -> void
foreign import ccall "THStorage.h clearFlag"
  c_clearFlag :: Ptr CTHFloatStorage -> CChar -> IO ()

-- | c_retain : storage -> void
foreign import ccall "THStorage.h retain"
  c_retain :: Ptr CTHFloatStorage -> IO ()

-- | c_swap : storage1 storage2 -> void
foreign import ccall "THStorage.h swap"
  c_swap :: Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ()

-- | c_free : storage -> void
foreign import ccall "THStorage.h free"
  c_free :: Ptr CTHFloatStorage -> IO ()

-- | c_resize : storage size -> void
foreign import ccall "THStorage.h resize"
  c_resize :: Ptr CTHFloatStorage -> CPtrdiff -> IO ()

-- | c_fill : storage value -> void
foreign import ccall "THStorage.h fill"
  c_fill :: Ptr CTHFloatStorage -> CFloat -> IO ()

-- |p_data : Pointer to function :  -> real *
foreign import ccall "THStorage.h &data"
  p_data :: FunPtr (Ptr CTHFloatStorage -> IO (Ptr CFloat))

-- |p_size : Pointer to function :  -> ptrdiff_t
foreign import ccall "THStorage.h &size"
  p_size :: FunPtr (Ptr CTHFloatStorage -> CPtrdiff)

-- |p_elementSize : Pointer to function :  -> size_t
foreign import ccall "THStorage.h &elementSize"
  p_elementSize :: FunPtr (CSize)

-- |p_set : Pointer to function :    -> void
foreign import ccall "THStorage.h &set"
  p_set :: FunPtr (Ptr CTHFloatStorage -> CPtrdiff -> CFloat -> IO ())

-- |p_get : Pointer to function :   -> real
foreign import ccall "THStorage.h &get"
  p_get :: FunPtr (Ptr CTHFloatStorage -> CPtrdiff -> CFloat)

-- |p_new : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &new"
  p_new :: FunPtr (IO (Ptr CTHFloatStorage))

-- |p_newWithSize : Pointer to function : size -> THStorage *
foreign import ccall "THStorage.h &newWithSize"
  p_newWithSize :: FunPtr (CPtrdiff -> IO (Ptr CTHFloatStorage))

-- |p_newWithSize1 : Pointer to function :  -> THStorage *
foreign import ccall "THStorage.h &newWithSize1"
  p_newWithSize1 :: FunPtr (CFloat -> IO (Ptr CTHFloatStorage))

-- |p_newWithSize2 : Pointer to function :   -> THStorage *
foreign import ccall "THStorage.h &newWithSize2"
  p_newWithSize2 :: FunPtr (CFloat -> CFloat -> IO (Ptr CTHFloatStorage))

-- |p_newWithSize3 : Pointer to function :    -> THStorage *
foreign import ccall "THStorage.h &newWithSize3"
  p_newWithSize3 :: FunPtr (CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage))

-- |p_newWithSize4 : Pointer to function :     -> THStorage *
foreign import ccall "THStorage.h &newWithSize4"
  p_newWithSize4 :: FunPtr (CFloat -> CFloat -> CFloat -> CFloat -> IO (Ptr CTHFloatStorage))

-- |p_newWithMapping : Pointer to function : filename size flags -> THStorage *
foreign import ccall "THStorage.h &newWithMapping"
  p_newWithMapping :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTHFloatStorage))

-- |p_newWithData : Pointer to function : data size -> THStorage *
foreign import ccall "THStorage.h &newWithData"
  p_newWithData :: FunPtr (Ptr CFloat -> CPtrdiff -> IO (Ptr CTHFloatStorage))

-- |p_newWithAllocator : Pointer to function : size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &newWithAllocator"
  p_newWithAllocator :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage))

-- |p_newWithDataAndAllocator : Pointer to function : data size allocator allocatorContext -> THStorage *
foreign import ccall "THStorage.h &newWithDataAndAllocator"
  p_newWithDataAndAllocator :: FunPtr (Ptr CFloat -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTHFloatStorage))

-- |p_setFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &setFlag"
  p_setFlag :: FunPtr (Ptr CTHFloatStorage -> CChar -> IO ())

-- |p_clearFlag : Pointer to function : storage flag -> void
foreign import ccall "THStorage.h &clearFlag"
  p_clearFlag :: FunPtr (Ptr CTHFloatStorage -> CChar -> IO ())

-- |p_retain : Pointer to function : storage -> void
foreign import ccall "THStorage.h &retain"
  p_retain :: FunPtr (Ptr CTHFloatStorage -> IO ())

-- |p_swap : Pointer to function : storage1 storage2 -> void
foreign import ccall "THStorage.h &swap"
  p_swap :: FunPtr (Ptr CTHFloatStorage -> Ptr CTHFloatStorage -> IO ())

-- |p_free : Pointer to function : storage -> void
foreign import ccall "THStorage.h &free"
  p_free :: FunPtr (Ptr CTHFloatStorage -> IO ())

-- |p_resize : Pointer to function : storage size -> void
foreign import ccall "THStorage.h &resize"
  p_resize :: FunPtr (Ptr CTHFloatStorage -> CPtrdiff -> IO ())

-- |p_fill : Pointer to function : storage value -> void
foreign import ccall "THStorage.h &fill"
  p_fill :: FunPtr (Ptr CTHFloatStorage -> CFloat -> IO ())